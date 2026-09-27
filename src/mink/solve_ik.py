"""Build and solve the inverse kinematics problem."""

from typing import Sequence

import numpy as np
import qpsolvers

from .configuration import Configuration
from .elastic import Elastic
from .exceptions import NoSolutionFound
from .limits import Limit
from .tasks import BaseTask, Objective, Task


def _compute_qp_objective(
    configuration: Configuration, tasks: Sequence[BaseTask], damping: float
) -> Objective:
    r"""Assemble the QP objective :math:`(H, c)` from all tasks.

    Tasks whose Hessian has the low-rank form :math:`J^T J` expose a weighted
    residual :math:`(W_i, e_i, \mu_i)`; stacking the :math:`W_i` lets us compute
    :math:`\sum_i W_i^T W_i = W^T W` with a single matrix multiply rather than
    summing per-task Hessians. Per-task Levenberg-Marquardt terms :math:`\mu_i`
    sum into the diagonal alongside the global ``damping``. Any task that returns
    no residual (e.g. an inertia-weighted Hessian) is added densely.
    """
    nv = configuration.model.nv

    weighted_jacobians: list[np.ndarray] = []
    weighted_errors: list[np.ndarray] = []
    mu_total = 0.0
    H_dense: np.ndarray | None = None
    c_dense: np.ndarray | None = None
    for task in tasks:
        residual = task.compute_qp_residual(configuration)
        if residual is None:
            H_task, c_task = task.compute_qp_objective(configuration)
            H_dense = H_task if H_dense is None else H_dense + H_task
            c_dense = c_task if c_dense is None else c_dense + c_task
        else:
            weighted_jacobian, weighted_error, mu = residual
            weighted_jacobians.append(weighted_jacobian)
            weighted_errors.append(weighted_error)
            mu_total += mu

    if weighted_jacobians:
        W = np.vstack(weighted_jacobians)
        H = W.T @ W
        c = -(np.concatenate(weighted_errors) @ W)
    else:
        H = np.zeros((nv, nv))
        c = np.zeros(nv)

    # Global LM damping plus the summed per-task LM terms on the diagonal.
    H.flat[:: nv + 1] += damping + mu_total

    if H_dense is not None:
        assert c_dense is not None  # Set together with H_dense.
        H += H_dense
        c += c_dense
    return Objective(H, c)


def _compute_qp_inequalities(
    configuration: Configuration, limits: Sequence[Limit] | None, dt: float
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if limits is None:
        limits = [configuration._default_limit]
    G_list: list[np.ndarray] = []
    h_list: list[np.ndarray] = []
    for limit in limits:
        inequality = limit.compute_qp_inequalities(configuration, dt)
        if not inequality.inactive:
            assert inequality.G is not None and inequality.h is not None
            G_list.append(inequality.G)
            h_list.append(inequality.h)
    if not G_list:
        return None, None
    return np.vstack(G_list), np.hstack(h_list)


def _compute_qp_equalities(
    configuration: Configuration,
    constraints: Sequence[Task] | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not constraints:
        return None, None
    A_list = []
    b_list = []
    for task in constraints:
        jacobian = task.compute_jacobian(configuration)
        feedback = -task.gain * task.compute_error(configuration)
        A_list.append(jacobian)
        b_list.append(feedback)
    return np.vstack(A_list), np.hstack(b_list)


def _compute_qp(
    configuration: Configuration,
    tasks: Sequence[BaseTask],
    dt: float,
    damping: float,
    limits: Sequence[Limit] | None,
    constraints: Sequence[Task | Elastic] | None,
) -> tuple[qpsolvers.Problem, np.ndarray]:
    """Assemble the QP with every constraint held exactly.

    Rows of elastic constraints come last in :math:`(A, b)`. Returns the problem
    and the penalty :math:`\rho` of those rows.
    """
    hard: list[Task] = []
    elastic: list[Elastic] = []
    for constraint in constraints or ():
        if isinstance(constraint, Elastic):
            elastic.append(constraint)
        else:
            hard.append(constraint)
    H, c = _compute_qp_objective(configuration, tasks, damping)
    G, h = _compute_qp_inequalities(configuration, limits, dt)
    A, b = _compute_qp_equalities(configuration, hard)
    rho = np.empty(0)
    if elastic:
        rows = [
            constraint.compute_penalized_rows(configuration) for constraint in elastic
        ]
        rho = np.concatenate([row[2] for row in rows])
        A_list = (
            [row[0] for row in rows] if A is None else [A] + [row[0] for row in rows]
        )
        b_list = (
            [-row[1] for row in rows] if b is None else [b] + [-row[1] for row in rows]
        )
        A, b = np.concatenate(A_list), np.concatenate(b_list)
    return qpsolvers.Problem(H, c, G, h, A, b), rho


def _soften(problem: qpsolvers.Problem, rho: np.ndarray) -> qpsolvers.Problem:
    r"""Replace the last ``len(rho)`` equality rows :math:`J \Delta q = -\alpha e`
    by :math:`\rho \odot (J \Delta q + \alpha e) = u - w`, with :math:`u, w \geq 0`
    and cost :math:`1^T (u + w)`, over :math:`x = [\Delta q; u; w]`."""
    H, c, G, h, A, b = problem.unpack_as_dense()[:6]
    assert A is not None and b is not None
    m, nv = rho.size, H.shape[0]
    n = nv + 2 * m
    H_x = np.zeros((n, n))
    H_x[:nv, :nv] = H
    c_x = np.ones(n)
    c_x[:nv] = c
    G_x = None
    if G is not None:
        G_x = np.zeros((G.shape[0], n))
        G_x[:, :nv] = G
    A_x = np.zeros((A.shape[0], n))
    A_x[:, :nv] = A
    b_x = b.copy()
    k = A.shape[0] - m
    A_x[k:, :nv] *= rho[:, None]
    b_x[k:] *= rho
    i = np.arange(m)
    A_x[k + i, nv + i] = -1.0
    A_x[k + i, nv + m + i] = 1.0
    lb = np.full(n, -np.inf)
    lb[nv:] = 0.0
    return qpsolvers.Problem(H_x, c_x, G_x, h, A_x, b_x, lb=lb)


def build_ik(
    configuration: Configuration,
    tasks: Sequence[BaseTask],
    dt: float,
    damping: float = 1e-12,
    limits: Sequence[Limit] | None = None,
    constraints: Sequence[Task | Elastic] | None = None,
) -> qpsolvers.Problem:
    r"""Build the quadratic program given the current configuration and tasks.

    The quadratic program is defined as:

    .. math::

        \begin{align*}
            \min_{\Delta q} & \frac{1}{2} \Delta q^T H \Delta q + c^T \Delta q \\
            \text{s.t.} \quad & G \Delta q \leq h \\
            & A \Delta q = b
        \end{align*}

    where :math:`v = \Delta q / dt` is the velocity in tangent space.

    Each penalized component :math:`i` of an :class:`~mink.Elastic` constraint
    adds variables :math:`u_i, w_i \geq 0` with cost :math:`u_i + w_i` and the row
    :math:`\rho_i (J_i \Delta q + \alpha e_i) = u_i - w_i`, so that at the optimum
    :math:`u_i + w_i = \rho_i |J_i \Delta q + \alpha e_i|`. The decision variable
    is then :math:`x = [\Delta q; u; w]`.

    Args:
        configuration: Robot configuration.
        tasks: List of kinematic tasks.
        dt: Integration timestep in [s].
        damping: Levenberg-Marquardt damping. Higher values improve numerical
            stability but slow down task convergence. This value applies to all
            dofs, including floating-base coordinates.
        limits: List of limits to enforce. Set to empty list to disable. If None,
            defaults to a configuration limit.
        constraints: List of tasks to enforce as equality constraints. These tasks
            will be satisfied exactly rather than in a least-squares sense. Wrap a
            task in :class:`~mink.Elastic` to let it yield when it cannot be met.

    Returns:
        Quadratic program of the inverse kinematics problem. With elastic
        constraints, its first :math:`n_v` variables are :math:`\Delta q`.
    """
    problem, rho = _compute_qp(configuration, tasks, dt, damping, limits, constraints)
    return _soften(problem, rho) if rho.size else problem


def solve_ik(
    configuration: Configuration,
    tasks: Sequence[BaseTask],
    dt: float,
    solver: str,
    damping: float = 1e-12,
    safety_break: bool = False,
    limits: Sequence[Limit] | None = None,
    constraints: Sequence[Task | Elastic] | None = None,
    **kwargs,
) -> np.ndarray:
    r"""Solve the differential inverse kinematics problem.

    Computes a velocity tangent to the current robot configuration. The computed
    velocity satisfies at (weighted) best the set of provided kinematic tasks.

    With :class:`~mink.Elastic` constraints, the QP is first solved with them held
    exactly. If every multiplier satisfies :math:`|\lambda_i| < \rho_i`, that
    solution is also the elastic one. Otherwise the penalized problem of
    :func:`build_ik` is solved, as in the elastic mode of [SNOPT]_.

    Args:
        configuration: Robot configuration.
        tasks: List of kinematic tasks.
        dt: Integration timestep in [s].
        solver: Backend quadratic programming (QP) solver.
        damping: Levenberg-Marquardt damping applied to all tasks. Higher values
            improve numerical stability but slow down task convergence. This
            value applies to all dofs, including floating-base coordinates.
        safety_break: If True, stop execution and raise an exception if
            the current configuration is outside limits. If False, print a
            warning and continue execution.
        limits: List of limits to enforce. Set to empty list to disable. If None,
            defaults to a configuration limit.
        constraints: List of tasks to enforce as equality constraints. These tasks
            will be satisfied exactly rather than in a least-squares sense. Wrap a
            task in :class:`~mink.Elastic` to let it yield when it cannot be met.
        kwargs: Keyword arguments to forward to the backend QP solver.

    Raises:
        NotWithinConfigurationLimits: If the current configuration is outside
            the joint limits and `safety_break` is True.
        NoSolutionFound: If the QP solver fails to find a solution.

    Returns:
        Velocity :math:`v` in tangent space.
    """
    configuration.check_limits(safety_break=safety_break)
    problem, rho = _compute_qp(configuration, tasks, dt, damping, limits, constraints)
    result = qpsolvers.solve_problem(problem, solver=solver, **kwargs)
    if rho.size and not (
        result.found
        and result.y is not None
        and np.all(np.abs(result.y[-rho.size :]) < rho)
    ):
        result = qpsolvers.solve_problem(_soften(problem, rho), solver=solver, **kwargs)
    if not result.found:
        raise NoSolutionFound(solver)
    assert result.x is not None
    delta_q = result.x[: configuration.nv]
    v: np.ndarray = delta_q / dt
    return v
