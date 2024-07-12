"""Build and solve the inverse kinematics problem."""

from typing import Sequence

import numpy as np
import qpsolvers

from mink.configuration import Configuration
from mink.limits import Constraint, Limit
from mink.tasks import Objective, Task


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
    configuration: Configuration, limits: Sequence[Limit], dt: float
) -> Constraint:
    G_list = []
    h_list = []
    for limit in limits:
        inequality = limit.compute_qp_inequalities(configuration, dt)
        G_list.append(inequality.G)
        h_list.append(inequality.h)
    if not G_list:
        return None, None
    return np.vstack(G_list), np.hstack(h_list)


def build_ik(
    configuration: Configuration,
    tasks: Sequence[Task],
    limits: Sequence[Limit],
    dt: float,
    damping: float = 1e-12,
) -> qpsolvers.Problem:
    P, q = _compute_qp_objective(configuration, tasks, damping)
    G, h = _compute_qp_inequalities(configuration, limits, dt)
    return qpsolvers.Problem(P, q, G, h)


def solve_ik(
    configuration: Configuration,
    tasks: Sequence[Task],
    limits: Sequence[Limit],
    dt: float,
    solver: str,
    damping: float = 1e-12,
    **kwargs,
) -> np.ndarray:
    """Compute a velocity tangent to the current configuration."""
    problem = build_ik(configuration, tasks, limits, dt, damping)
    result = qpsolvers.solve_problem(problem, solver=solver, **kwargs)
    dq = result.x
    assert dq is not None
    v: np.ndarray = dq / dt
    return v
