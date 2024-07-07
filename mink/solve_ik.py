"""Build and solve the inverse kinematics problem."""

from __future__ import annotations

from typing import Sequence
import numpy as np

from mink.configuration import Configuration
from mink.tasks import Task, Objective
from mink.limits import Limit, BoxConstraint, CollisionAvoidanceLimit

import mujoco
from dataclasses import dataclass
# import qpsolvers


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
        # configuration: Configuration,
        n: int,
        H: np.ndarray,
        c: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        prev_sol: np.ndarray | None,
    ) -> Problem:
        # n = configuration.model.nv
        dq = np.zeros(n) if prev_sol is None else prev_sol
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
    q = configuration.q
    lower_limits = []
    upper_limits = []
    for limit in limits:
        if isinstance(limit, CollisionAvoidanceLimit):
            continue
        inequality = limit.compute_qp_inequalities(q, dt)
        if inequality.inactive():
            continue
        lower_limits.append(inequality.lower)
        upper_limits.append(inequality.upper)
    if not lower_limits:
        return BoxConstraint()
    lower = np.maximum.reduce(lower_limits)
    upper = np.minimum.reduce(upper_limits)
    return BoxConstraint(lower, upper)


# def build_ik_dual(
#     configuration: Configuration,
#     tasks: Sequence[Task],
#     limits: Sequence[Limit],
#     dt: float,
#     damping: float = 1e-12,
#     prev_sol: np.ndarray | None = None,
# ) -> Problem:
#     """Build a Quadratic Program (QP) for the current configuration and tasks."""
#     H, c = _compute_qp_objective(configuration, tasks, damping)
#     limits_subset = [
#         limit for limit in limits if not isinstance(limit, CollisionAvoidanceLimit)
#     ]
#     lower, upper = _compute_qp_inequalities(configuration, limits_subset, dt)
#     limit = None
#     for l in limits:
#         if isinstance(l, CollisionAvoidanceLimit):
#             limit = l
#             break
#     assert limit is not None
#     _, D, N = limit.compute_qp_inequalities(configuration.data, configuration.q, dt)
#     N = np.vstack([N, np.eye(upper.shape[0], c.shape[0]), -np.eye(upper.shape[0], c.shape[0])])
#     D = np.hstack([D, upper, -lower])
#     H_inv = np.linalg.pinv(H)
#     Q = N @ H_inv @ N.T
#     b = N @ H_inv.T @ c + D
#     low = np.zeros((b.shape[0],))
#     problem = Problem.initialize(b.shape[0], Q, b, low, np.full_like(low, np.inf), prev_sol)
#     lam = problem.solve()
#     dq = -H_inv @ (c + N.T @ lam)
#     return dq, lam


def build_ik(
    configuration: Configuration,
    tasks: Sequence[Task],
    limits: Sequence[Limit],
    dt: float,
    damping: float = 1e-12,
    prev_sol: np.ndarray | None = None,
) -> Problem:
    """Build a Quadratic Program (QP) for the current configuration and tasks."""
    H, c = _compute_qp_objective(configuration, tasks, damping)
    lower, upper = _compute_qp_inequalities(configuration, limits, dt)

    limit = None
    for l in limits:
        if isinstance(l, CollisionAvoidanceLimit):
            limit = l
            break
    assert limit is not None
    A, b = limit.compute_qp_inequalities(configuration.data, configuration.q, dt)

    def compute_qp_modifications(Q, c, A, b, lam):
        m = A.shape[0]
        Q_prime = Q.copy()
        c_prime = c.copy()
        for i in range(m):
            if np.isfinite(b[i]):
                a_i_outer = np.outer(A[i], A[i])
                Q_prime += lam * a_i_outer
                c_prime -= 2 * lam * b[i] * A[i]
        return Q_prime, c_prime

    lam = 1e-3
    lambda_max = 1e4
    successful = False
    tolerance = 1e-8
    lambda_increment_factor = 10
    while lam <= lambda_max:
        H_prime, c_prime = compute_qp_modifications(H, c, A, b, lam)
        problem = Problem.initialize(
            n=H_prime.shape[0],
            H=H_prime,
            c=c_prime,
            lower=lower,
            upper=upper,
            prev_sol=prev_sol
        )
        solution = problem.solve()
        constraint_violations = np.maximum(0, A @ solution - b)
        max_violation = np.max(constraint_violations)
        if max_violation < tolerance:
            successful = True
            print(f"Constraints are satisfactorily met")
            break
        lam *= lambda_increment_factor
    if not successful:
        raise RuntimeError("Failed to meet constraints within the lambda range.")
    return solution


def solve_ik(
    configuration: Configuration,
    tasks: Sequence[Task],
    limits: Sequence[Limit],
    dt: float,
    # solver: str,
    damping: float = 1e-12,
    prev_sol: np.ndarray | None = None,
    # **kwargs,
) -> np.ndarray:
    """Compute a velocity tangent to the current configuration."""
    # dq, lam = build_ik(configuration, tasks, limits, dt, damping, prev_sol)
    # result = qpsolvers.solve_problem(problem, solver=solver, **kwargs)
    # dq = result.x
    # assert dq is not None
    # return dq / dt, lam
    dq = build_ik(configuration, tasks, limits, dt, damping, prev_sol)
    # dq = problem.solve()
    return dq / dt
