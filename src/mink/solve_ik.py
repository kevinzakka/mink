"""Build and solve the inverse kinematics problem."""

from typing import Literal, Sequence

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
) -> tuple[qpsolvers.Problem, np.ndarray, np.ndarray]:
    r"""Assemble the QP with every constraint held exactly.

    Rows of elastic constraints come last in :math:`(A, b)`. Returns the problem
    and the penalty :math:`\rho` and penalty split :math:`p` of those rows.
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
    split = np.empty(0)
    if elastic:
        rows = [
            constraint.compute_penalized_rows(configuration) for constraint in elastic
        ]
        rho = np.concatenate([row[2] for row in rows])
        split = np.concatenate(
            [
                np.full(row[2].size, constraint.penalty_split)
                for row, constraint in zip(rows, elastic, strict=True)
            ]
        )
        A_list = (
            [row[0] for row in rows] if A is None else [A] + [row[0] for row in rows]
        )
        b_list = (
            [-row[1] for row in rows] if b is None else [b] + [-row[1] for row in rows]
        )
        A, b = np.concatenate(A_list), np.concatenate(b_list)
    return qpsolvers.Problem(H, c, G, h, A, b), rho, split


ElasticStrategy = Literal["hard_first", "penalty"]
"""How :func:`solve_ik` handles :class:`~mink.Elastic` constraints."""


def _soften(
    problem: qpsolvers.Problem, rho: np.ndarray, split: np.ndarray
) -> qpsolvers.Problem:
    r"""Replace the last ``len(rho)`` equality rows by their :math:`\ell_1` penalty.

    Each row :math:`J_i \Delta q = -\alpha e_i`, with residual
    :math:`r_i = J_i \Delta q + \alpha e_i`, gets a slack :math:`s_i` with cost
    :math:`\rho_i^{1-p} s_i` and the rows
    :math:`-s_i \leq \rho_i^p r_i \leq s_i`, over :math:`x = [\Delta q; s]`. At
    the optimum :math:`s_i = \rho_i^p |r_i|`, so the cost is
    :math:`\rho_i |r_i|` for any split :math:`p`. The split only changes the
    scaling: :math:`p = 1/2` keeps both the rows and the slack cost within
    :math:`\sqrt{\rho}` of unit scale, which keeps DAQP accurate for very small
    and very large penalties.
    """
    H, c, G, h, A, b = problem.unpack_as_dense()[:6]
    assert A is not None and b is not None
    m, nv = rho.size, H.shape[0]
    k = A.shape[0] - m
    H_x = np.zeros((nv + m, nv + m))
    H_x[:nv, :nv] = H
    c_x = np.concatenate([c, rho ** (1.0 - split)])
    scale = rho**split
    scaled_A = scale[:, None] * A[k:]
    scaled_b = scale * b[k:]
    neg_eye = -np.eye(m)
    G_x = np.block([[scaled_A, neg_eye], [-scaled_A, neg_eye]])
    h_x = np.concatenate([scaled_b, -scaled_b])
    if G is not None:
        assert h is not None
        G_x = np.vstack([np.hstack([G, np.zeros((G.shape[0], m))]), G_x])
        h_x = np.concatenate([h, h_x])
    A_x, b_x = None, None
    if k:
        A_x, b_x = np.hstack([A[:k], np.zeros((k, m))]), b[:k]
    return qpsolvers.Problem(H_x, c_x, G_x, h_x, A_x, b_x)


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

    Each penalized component :math:`i` of an :class:`~mink.Elastic` constraint,
    with residual :math:`r_i = J_i \Delta q + \alpha e_i`, penalty
    :math:`\rho_i` and penalty split :math:`p` (see :class:`~mink.Elastic`), adds a slack :math:`s_i` with cost :math:`\rho_i^{1-p} s_i`
    and the rows :math:`-s_i \leq \rho_i^p r_i \leq s_i`. At the optimum
    :math:`s_i = \rho_i^p |r_i|`, so the added cost is the :math:`\ell_1`
    penalty :math:`\rho_i |r_i|` for any split :math:`p`. The decision variable
    is then :math:`x = [\Delta q; s]`.

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
    problem, rho, split = _compute_qp(
        configuration, tasks, dt, damping, limits, constraints
    )
    return _soften(problem, rho, split) if rho.size else problem


def solve_ik(
    configuration: Configuration,
    tasks: Sequence[BaseTask],
    dt: float,
    solver: str,
    damping: float = 1e-12,
    safety_break: bool = False,
    limits: Sequence[Limit] | None = None,
    constraints: Sequence[Task | Elastic] | None = None,
    elastic_strategy: ElasticStrategy = "hard_first",
    **kwargs,
) -> np.ndarray:
    r"""Solve the differential inverse kinematics problem.

    Computes a velocity tangent to the current robot configuration. The computed
    velocity satisfies at (weighted) best the set of provided kinematic tasks.

    With :class:`~mink.Elastic` constraints and the default ``"hard_first"``
    strategy, the QP is first solved with them held exactly. If every multiplier
    satisfies :math:`|\lambda_i| < \rho_i`, that solution is also the elastic
    one. Otherwise the penalized problem of :func:`build_ik` is solved, as in the
    elastic mode of [SNOPT]_. The ``"penalty"`` strategy always solves the
    penalized problem directly: one QP per call, with the same solution up to
    solver accuracy.

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
        elastic_strategy: ``"hard_first"`` solves with elastic constraints held
            exactly and falls back to the penalized problem only when some
            constraint must yield: exact while held, but two QPs when yielding.
            ``"penalty"`` always solves the penalized problem: one QP per call.
        kwargs: Keyword arguments to forward to the backend QP solver.

    Raises:
        NotWithinConfigurationLimits: If the current configuration is outside
            the joint limits and `safety_break` is True.
        NoSolutionFound: If the QP solver fails to find a solution.
        ValueError: If ``elastic_strategy`` is not ``"hard_first"`` or
            ``"penalty"``.

    Returns:
        Velocity :math:`v` in tangent space.
    """
    if elastic_strategy not in ("hard_first", "penalty"):
        raise ValueError(
            "`elastic_strategy` must be 'hard_first' or 'penalty', got "
            f"{elastic_strategy!r}"
        )
    configuration.check_limits(safety_break=safety_break)
    problem, rho, split = _compute_qp(
        configuration, tasks, dt, damping, limits, constraints
    )
    if rho.size and elastic_strategy == "penalty":
        problem = _soften(problem, rho, split)
    result = qpsolvers.solve_problem(problem, solver=solver, **kwargs)
    if (
        rho.size
        and elastic_strategy == "hard_first"
        and not (
            result.found
            and result.y is not None
            and np.all(np.abs(result.y[-rho.size :]) < rho)
        )
    ):
        softened = _soften(problem, rho, split)
        result = qpsolvers.solve_problem(softened, solver=solver, **kwargs)
    if not result.found:
        raise NoSolutionFound(solver)
    assert result.x is not None
    delta_q = result.x[: configuration.nv]
    v: np.ndarray = delta_q / dt
    return v
