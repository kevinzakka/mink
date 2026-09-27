"""Build and solve the inverse kinematics problem."""

from typing import Sequence

import numpy as np
import qpsolvers

from .configuration import Configuration
from .exceptions import NoSolutionFound
from .limits import ConfigurationLimit, Limit
from .tasks import BaseTask, Objective, Task


def _is_elastic(task: BaseTask) -> bool:
    return isinstance(task, Task) and task.elastic


def _compute_qp_objective(
    configuration: Configuration, tasks: Sequence[BaseTask], damping: float
) -> Objective:
    r"""Assemble the QP objective :math:`(H, c)` from all tasks.

    Tasks whose Hessian has the low-rank form :math:`J^T J` expose a weighted
    residual :math:`(W_i, e_i, \mu_i)`; stacking the :math:`W_i` lets us compute
    :math:`\sum_i W_i^T W_i = W^T W` with a single matrix multiply rather than
    summing per-task Hessians. Per-task Levenberg-Marquardt terms :math:`\mu_i`
    sum into the diagonal alongside the global ``damping``. Any task that returns
    no residual (e.g. an inertia-weighted Hessian) is added densely. Elastic tasks
    are skipped here; see :func:`_compute_qp_elastic`.
    """
    nv = configuration.model.nv

    weighted_jacobians: list[np.ndarray] = []
    weighted_errors: list[np.ndarray] = []
    mu_total = 0.0
    H_dense: np.ndarray | None = None
    c_dense: np.ndarray | None = None
    for task in tasks:
        if _is_elastic(task):
            continue
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


def _compute_qp_elastic(
    configuration: Configuration, tasks: Sequence[BaseTask]
) -> tuple[np.ndarray, np.ndarray] | None:
    r"""Assemble the slack rows of all elastic tasks.

    Stacking the weighted residuals :math:`r = W J \Delta q - \bar{e}` of all
    elastic tasks (with :math:`\bar{e} = -\alpha W e`) and their penalties
    :math:`\rho`, and giving each task component its own slack :math:`s_i`, the
    :math:`\ell_1` penalty :math:`\rho^T |r|` is the linear cost :math:`1^T s`
    subject to :math:`-s \leq \rho \odot r \leq s`. At the optimum
    :math:`s = \rho \odot |r|` is the penalty paid by each component.

    Folding :math:`\rho` into the rows, rather than using the slack :math:`|r|`
    with cost :math:`\rho`, keeps the slacks on the same scale as the rest of
    the problem: solvers that regularize the (zero) slack Hessian, such as
    DAQP's proximal iterations, then stay accurate for large penalties.

    Returns ``(G, h)`` such that these rows read :math:`G [\Delta q; s] \leq h`,
    or ``None`` if no task is elastic. Each task's :math:`-I` block sits only in
    its own slack columns.
    """
    scaled_jacobians: list[np.ndarray] = []
    scaled_errors: list[np.ndarray] = []
    for task in tasks:
        if not _is_elastic(task):
            continue
        assert isinstance(task, Task)
        weighted_jacobian, weighted_error, _ = task.compute_qp_residual(configuration)
        penalty = task._penalty_vector(weighted_error.shape[0])
        scaled_jacobians.append(penalty[:, None] * weighted_jacobian)
        scaled_errors.append(penalty * weighted_error)

    if not scaled_jacobians:
        return None

    PWJ = np.vstack(scaled_jacobians)
    Pe = np.concatenate(scaled_errors)
    neg_eye = -np.eye(Pe.shape[0])
    G = np.block([[PWJ, neg_eye], [-PWJ, neg_eye]])
    h = np.concatenate([Pe, -Pe])
    return G, h


def _compute_qp_inequalities(
    configuration: Configuration, limits: Sequence[Limit] | None, dt: float
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if limits is None:
        limits = [ConfigurationLimit(configuration.model)]
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


def build_ik(
    configuration: Configuration,
    tasks: Sequence[BaseTask],
    dt: float,
    damping: float = 1e-12,
    limits: Sequence[Limit] | None = None,
    constraints: Sequence[Task] | None = None,
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

    If some tasks are elastic (see :class:`~mink.Task`), the decision variable is
    augmented with one slack variable per elastic task component,
    :math:`x = [\Delta q; s] \in \mathbb{R}^{n_v + m}` with :math:`m` the summed
    dimension of the elastic tasks, and the program becomes:

    .. math::

        \begin{align*}
            \min_{\Delta q, s} & \frac{1}{2} \Delta q^T H \Delta q + c^T \Delta q
                + 1^T s \\
            \text{s.t.} \quad & G \Delta q \leq h \\
            & A \Delta q = b \\
            & -s \leq \rho \odot W (J \Delta q + \alpha e) \leq s
        \end{align*}

    where :math:`H` and :math:`c` exclude the elastic tasks, and the last row
    stacks the weighted residuals of all elastic tasks, scaled by their penalties
    :math:`\rho`. At the optimum, :math:`1^T s` equals the :math:`\ell_1` penalty
    :math:`\rho^T |W (J \Delta q + \alpha e)|`. The first :math:`n_v` entries of
    the solution are :math:`\Delta q`.

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
            will be satisfied exactly rather than in a least-squares sense.

    Returns:
        Quadratic program of the inverse kinematics problem.
    """
    H, c = _compute_qp_objective(configuration, tasks, damping)
    G, h = _compute_qp_inequalities(configuration, limits, dt)
    A, b = _compute_qp_equalities(configuration, constraints)
    elastic = _compute_qp_elastic(configuration, tasks)
    if elastic is None:
        return qpsolvers.Problem(H, c, G, h, A, b)

    G_elastic, h_elastic = elastic
    nv = H.shape[0]
    m = G_elastic.shape[1] - nv
    H_aug = np.zeros((nv + m, nv + m))
    H_aug[:nv, :nv] = H
    c_aug = np.concatenate([c, np.ones(m)])
    if G is None:
        G_aug, h_aug = G_elastic, h_elastic
    else:
        assert h is not None
        G_aug = np.vstack([np.hstack([G, np.zeros((G.shape[0], m))]), G_elastic])
        h_aug = np.concatenate([h, h_elastic])
    A_aug = None if A is None else np.hstack([A, np.zeros((A.shape[0], m))])
    return qpsolvers.Problem(H_aug, c_aug, G_aug, h_aug, A_aug, b)


def solve_ik(
    configuration: Configuration,
    tasks: Sequence[BaseTask],
    dt: float,
    solver: str,
    damping: float = 1e-12,
    safety_break: bool = False,
    limits: Sequence[Limit] | None = None,
    constraints: Sequence[Task] | None = None,
    **kwargs,
) -> np.ndarray:
    r"""Solve the differential inverse kinematics problem.

    Computes a velocity tangent to the current robot configuration. The computed
    velocity satisfies at (weighted) best the set of provided kinematic tasks.
    Elastic tasks add slack variables to the QP (see :func:`build_ik`); only the
    velocity part of the solution is returned.

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
            will be satisfied exactly rather than in a least-squares sense.
        kwargs: Keyword arguments to forward to the backend QP solver.

    Raises:
        NotWithinConfigurationLimits: If the current configuration is outside
            the joint limits and `safety_break` is True.
        NoSolutionFound: If the QP solver fails to find a solution.

    Returns:
        Velocity :math:`v` in tangent space.
    """
    configuration.check_limits(safety_break=safety_break)
    problem = build_ik(configuration, tasks, dt, damping, limits, constraints)
    result = qpsolvers.solve_problem(problem, solver=solver, **kwargs)
    if not result.found:
        raise NoSolutionFound(solver)
    assert result.x is not None
    delta_q = result.x[: configuration.model.nv]
    v: np.ndarray = delta_q / dt
    return v
