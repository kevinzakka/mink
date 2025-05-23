"""Build and solve the inverse kinematics problem."""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import qpsolvers

from .configuration import Configuration
from .exceptions import NoSolutionFound
from .limits import ConfigurationLimit, Limit
from .tasks import BaseTask, Objective


def _compute_qp_objective(
    configuration: Configuration, tasks: Sequence[BaseTask], damping: float
) -> Objective:
    H = np.eye(configuration.model.nv) * damping
    c = np.zeros(configuration.model.nv)
    for task in tasks:
        H_task, c_task = task.compute_qp_objective(configuration)
        H += H_task
        c += c_task
    return Objective(H, c)


def _compute_qp_inequalities(
    configuration: Configuration, limits: Optional[Sequence[Limit]], dt: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if limits is None:
        limits = [ConfigurationLimit(configuration.model)]
    G_list: List[np.ndarray] = []
    h_list: List[np.ndarray] = []
    for limit in limits:
        inequality = limit.compute_qp_inequalities(configuration, dt)
        if not inequality.inactive:
            assert inequality.G is not None and inequality.h is not None  # mypy.
            G_list.append(inequality.G)
            h_list.append(inequality.h)
    if not G_list:
        return None, None
    return np.vstack(G_list), np.hstack(h_list)


def build_ik(
    configuration: Configuration,
    tasks: Sequence[BaseTask],
    dt: float,
    damping: float = 1e-12,
    limits: Optional[Sequence[Limit]] = None,
) -> qpsolvers.Problem:
    r"""Build quadratic program from current configuration and tasks.

    The quadratic program is defined as:

    .. math::

        \begin{align*}
            \min_{\Delta q} & \frac{1}{2} \Delta q^T H \Delta q + c^T \Delta q \\
            \text{s.t.} \quad & G \Delta q \leq h
        \end{align*}

    where :math:`\Delta q = v / dt` is the vector of joint displacements.

    Args:
        configuration: Robot configuration.
        tasks: List of kinematic tasks.
        dt: Integration timestep in [s].
        damping: Levenberg-Marquardt damping. Higher values improve numerical
            stability but slow down task convergence. This value applies to all
            dofs, including floating-base coordinates.
        limits: List of limits to enforce. Set to empty list to disable. If None,
            defaults to a configuration limit.

    Returns:
        Quadratic program of the inverse kinematics problem.
    """
    P, q = _compute_qp_objective(configuration, tasks, damping)
    G, h = _compute_qp_inequalities(configuration, limits, dt)
    return qpsolvers.Problem(P, q, G, h)


def solve_ik(
    configuration: Configuration,
    tasks: Sequence[BaseTask],
    dt: float,
    solver: str,
    damping: float = 1e-12,
    safety_break: bool = False,
    limits: Optional[Sequence[Limit]] = None,
    **kwargs,
) -> np.ndarray:
    r"""Solve the differential inverse kinematics problem.

    Computes a velocity tangent to the current robot configuration. The computed
    velocity satisfies at (weighted) best the set of provided kinematic tasks.

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
        kwargs: Keyword arguments to forward to the backend QP solver.

    Raises:
        NotWithinConfigurationLimits: If the current configuration is outside
            the joint limits and `safety_break` is True.
        NoSolutionFound: If the QP solver fails to find a solution.

    Returns:
        Velocity :math:`v` in tangent space.
    """
    configuration.check_limits(safety_break=safety_break)
    problem = build_ik(configuration, tasks, dt, damping, limits)
    result = qpsolvers.solve_problem(problem, solver=solver, **kwargs)
    if not result.found:
        raise NoSolutionFound(solver)
    delta_q = result.x
    assert delta_q is not None
    v: np.ndarray = delta_q / dt
    return v
