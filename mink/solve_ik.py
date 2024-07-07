"""Build and solve the inverse kinematics problem."""

from __future__ import annotations

from typing import Sequence
import numpy as np

from mink.configuration import Configuration
from mink.tasks import Task, Objective
from mink.limits import Limit, Constraint, CollisionAvoidanceLimit

# import mujoco
# from dataclasses import dataclass
import qpsolvers


class IKFailure(Exception):
    """Raised when the inverse kinematics problem cannot be solved."""


# @dataclass(frozen=True)
# class Problem:
#     H: np.ndarray
#     c: np.ndarray
#     lower: np.ndarray
#     upper: np.ndarray
#     n: int
#     dq: np.ndarray
#     R: np.ndarray
#     index: np.ndarray

#     @staticmethod
#     def initialize(
#         configuration: Configuration,
#         H: np.ndarray,
#         c: np.ndarray,
#         lower: np.ndarray,
#         upper: np.ndarray,
#         prev_sol: np.ndarray | None,
#     ) -> Problem:
#         n = configuration.model.nv
#         dq = np.zeros(n) if prev_sol is None else prev_sol
#         R = np.zeros((n, n + 7))
#         index = np.zeros(n, np.int32)
#         return Problem(H, c, lower, upper, n, dq, R, index)

#     def solve(self) -> np.ndarray:
#         rank = mujoco.mju_boxQP(
#             res=self.dq,
#             R=self.R,
#             index=self.index,
#             H=self.H,
#             g=self.c,
#             lower=self.lower,
#             upper=self.upper,
#         )
#         if rank == -1:
#             raise IKFailure("QP solver failed")
#         return self.dq


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
) -> Constraint:
    """Compute the box constraints for the inverse kinematics problem.

    The box constraints are of the form:

        lower <= x <= upper

    where x is the output of inverse kinematics.

    Args:
        configuration: The current configuration of the robot.
        dt: The integration time step in seconds.

    Returns:
        The box constraints.
    """
    q = configuration.q
    # lower_limits = []
    # upper_limits = []
    G_list = []
    h_list = []
    for limit in limits:
        if isinstance(limit, CollisionAvoidanceLimit):
            inequality = limit.compute_qp_inequalities(configuration.data, q, dt)
        else:
            inequality = limit.compute_qp_inequalities(q, dt)
        # if inequality.inactive():
        # continue
        # lower_limits.append(inequality.lower)
        # upper_limits.append(inequality.upper)
        G_list.append(inequality.G)
        h_list.append(inequality.h)
    # if not lower_limits:
    # return BoxConstraint()
    if not G_list:
        return None, None
    # lower = np.maximum.reduce(lower_limits)
    # upper = np.minimum.reduce(upper_limits)
    # return BoxConstraint(lower, upper)
    return np.vstack(G_list), np.hstack(h_list)


def build_ik(
    configuration: Configuration,
    tasks: Sequence[Task],
    limits: Sequence[Limit],
    dt: float,
    damping: float = 1e-12,
    # prev_sol: np.ndarray | None = None,
) -> qpsolvers.Problem:
    """Build a Quadratic Program (QP) for the current configuration and tasks."""
    H, c = _compute_qp_objective(configuration, tasks, damping)
    # lower, upper = _compute_qp_inequalities(configuration, limits, dt)
    G, h = _compute_qp_inequalities(configuration, limits, dt)
    # from ipdb import set_trace; set_trace()
    problem = qpsolvers.Problem(H, c, G, h)
    return problem
    # return Problem.initialize(configuration, H, c, lower, upper, prev_sol)


def solve_ik(
    configuration: Configuration,
    tasks: Sequence[Task],
    limits: Sequence[Limit],
    dt: float,
    solver: str,
    damping: float = 1e-12,
    # prev_sol: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """Compute a velocity tangent to the current configuration."""
    problem = build_ik(configuration, tasks, limits, dt, damping)
    result = qpsolvers.solve_problem(problem, solver=solver, **kwargs)
    # dq = problem.solve()
    dq = result.x
    assert dq is not None
    return dq / dt
