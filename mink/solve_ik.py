"""Build and solve the inverse kinematics problem."""

from typing import Optional, Sequence

import numpy as np
import qpsolvers

from .configuration import Configuration
from .limits import ConfigurationLimit, Limit
from .tasks import Objective, Task


def _compute_qp_objective(
    configuration: Configuration, tasks: Sequence[Task], damping: float
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
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if limits is None:
        limits = [ConfigurationLimit(configuration.model)]
    G_list: list[np.ndarray] = []
    h_list: list[np.ndarray] = []
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
    tasks: Sequence[Task],
    dt: float,
    damping: float = 1e-12,
    limits: Optional[Sequence[Limit]] = None,
) -> qpsolvers.Problem:
    """Build quadratic program from current configuration and tasks.

    Args:
        configuration: Robot configuration.
        tasks: List of kinematic tasks.
        dt: Integration timestep in [s].
        damping: Levenberg-Marquardt damping.
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
    tasks: Sequence[Task],
    dt: float,
    solver: str,
    damping: float = 1e-12,
    safety_break: bool = False,
    limits: Optional[Sequence[Limit]] = None,
    **kwargs,
) -> np.ndarray:
    """Solve the differential inverse kinematics problem.

    Computes a velocity tangent to the current robot configuration. The computed
    velocity satisfies at (weighted) best the set of provided kinematic tasks.

    Args:
        configuration: Robot configuration.
        tasks: List of kinematic tasks.
        dt: Integration timestep in [s].
        solver: Backend quadratic programming (QP) solver.
        damping: Levenberg-Marquardt damping.
        safety_break: If True, stop execution and raise an exception if
            the current configuration is outside limits. If False, print a
            warning and continue execution.
        limits: List of limits to enforce. Set to empty list to disable. If None,
            defaults to a configuration limit.
        kwargs: Keyword arguments to forward to the backend QP solver.

    Returns:
        Velocity `v` in tangent space.
    """
    configuration.check_limits(safety_break=safety_break)
    problem = build_ik(configuration, tasks, dt, damping, limits)
    result = qpsolvers.solve_problem(problem, solver=solver, **kwargs)
    dq = result.x
    assert dq is not None
    v: np.ndarray = dq / dt
    return v
