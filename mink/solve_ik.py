"""Build and solve the inverse kinematics problem."""

from typing import Sequence
import numpy as np
import qpsolvers

from mink.configuration import Configuration
from mink.tasks import Task, Objective
from mink.limits import Limit, Inequality


def _compute_qp_objective(
    configuration: Configuration,
    tasks: Sequence[Task],
    damping: float,
) -> Objective:
    """Compute the quadratic objective function for the inverse kinematics problem.

    The Hessian matrix H and the linear term c define the QP objective as:

        0.5 x^T H x + c^T x

    where x is the output of inverse kinematics. We divide by dt to obtain a commanded
    velocity.

    Args:
        configuration: The current configuration of the robot.
        tasks: A sequence of kinematic tasks to fulfill.

    Returns:
        The quadratic objective function.
    """
    H = np.eye(configuration.model.nv) * damping
    c = np.zeros(configuration.model.nv)
    for task in tasks:
        H_task, c_task = task.compute_qp_objective(configuration)
        H += H_task
        c += c_task
    return Objective(H, c)


def _compute_qp_inequalities(
    configuration: Configuration,
    limits: Sequence[Limit],
    dt: float,
) -> Inequality:
    """Compute the inequality constraints for the inverse kinematics problem.

    The inequality constraints are affine functions of the joint velocities and are of
    the form:

        G x <= h

    where x is the output of inverse kinematics.

    Args:
        configuration: The current configuration of the robot.
        dt: The integration time step in seconds.

    Returns:
        The inequality constraints.
    """
    q = configuration.q
    G_list = []
    h_list = []
    for limit in limits:
        inequality = limit.compute_qp_inequalities(q, dt)
        if inequality.inactive():
            continue
        G_list.append(inequality.G)
        h_list.append(inequality.h)
    if not G_list:
        return Inequality()
    G = np.vstack(G_list)
    h = np.hstack(h_list)
    return Inequality(G, h)


def build_ik(
    configuration: Configuration,
    tasks: Sequence[Task],
    limits: Sequence[Limit],
    dt: float,
    damping: float = 1e-12,
) -> qpsolvers.Problem:
    """Build a Quadratic Program (QP) for the current configuration and tasks."""
    P, q = _compute_qp_objective(configuration, tasks, damping)
    G, h = _compute_qp_inequalities(configuration, limits, dt)
    problem = qpsolvers.Problem(P=P, q=q, G=G, h=h)
    return problem


def solve_ik(
    configuration: Configuration,
    tasks: Sequence[Task],
    limits: Sequence[Limit],
    dt: float,
    solver: str,
    **kwargs,
) -> np.ndarray:
    """Compute a velocity tangent to the current configuration."""
    problem = build_ik(configuration, tasks, limits, dt)
    result = qpsolvers.solve_problem(problem=problem, solver=solver, **kwargs)
    dq = result.x
    assert dq is not None
    return dq / dt
