"""Build and solve the inverse kinematics problem."""

from __future__ import annotations

from typing import Sequence
import numpy as np

from mink.configuration import Configuration
from mink.tasks import Task, Objective
from mink.limits import Limit, BoxConstraint

import mujoco
from dataclasses import dataclass


class IKFailure(Exception):
    """Raised when the inverse kinematics problem cannot be solved."""


@dataclass(frozen=True)
class Problem:
    H: np.ndarray
    c: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    n: int
    dq: np.ndarray
    R: np.ndarray
    index: np.ndarray

    @staticmethod
    def initialize(
        configuration: Configuration,
        H: np.ndarray,
        c: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> Problem:
        n = configuration.model.nv
        dq = np.zeros(n)
        R = np.zeros((n, n + 7))
        index = np.zeros(n, np.int32)
        return Problem(H, c, lower, upper, n, dq, R, index)

    def solve(self) -> np.ndarray:
        rank = mujoco.mju_boxQP(
            res=self.dq,
            R=self.R,
            index=self.index,
            H=self.H,
            g=self.c,
            lower=self.lower,
            upper=self.upper,
        )
        if rank == -1:
            raise IKFailure("QP solver failed")
        return self.dq


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
) -> BoxConstraint:
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
    lower_limits = []
    upper_limits = []
    for limit in limits:
        inequality = limit.compute_qp_inequalities(configuration.q, dt)
        if inequality.inactive():
            continue
        lower_limits.append(inequality.lower)
        upper_limits.append(inequality.upper)
    if not lower_limits:
        return BoxConstraint()
    lower = np.maximum.reduce(lower_limits)
    upper = np.minimum.reduce(upper_limits)
    return BoxConstraint(lower, upper)


def build_ik(
    configuration: Configuration,
    tasks: Sequence[Task],
    limits: Sequence[Limit],
    dt: float,
    damping: float = 1e-12,
) -> Problem:
    """Build a Quadratic Program (QP) for the current configuration and tasks."""
    H, c = _compute_qp_objective(configuration, tasks, damping)
    lower, upper = _compute_qp_inequalities(configuration, limits, dt)
    return Problem.initialize(configuration, H, c, lower, upper)


def solve_ik(
    configuration: Configuration,
    tasks: Sequence[Task],
    limits: Sequence[Limit],
    dt: float,
    damping: float = 1e-12,
) -> np.ndarray:
    """Compute a velocity tangent to the current configuration."""
    problem = build_ik(configuration, tasks, limits, dt, damping)
    dq = problem.solve()
    return dq / dt
