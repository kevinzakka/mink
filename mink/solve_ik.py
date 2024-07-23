"""Build and solve the inverse kinematics problem."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import mujoco

from .configuration import Configuration
from .limits import Limit
from .tasks import Objective, Task


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
        prev_sol: np.ndarray | None,
    ):
        n = configuration.model.nv
        dq = np.zeros(n) if prev_sol is None else prev_sol
        R = np.zeros((n, n + 7))
        index = np.zeros(n, np.int32)
        return Problem(H, c, lower, upper, n, dq, R, index)

    def solve(self) -> np.ndarray | None:
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
            return None
        return self.dq


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
) -> tuple[np.ndarray | None, np.ndarray | None]:
    lower_list = []
    upper_list = []
    for limit in limits:
        inequality = limit.compute_qp_inequalities(configuration, dt)
        if not inequality.inactive:
            assert inequality.G is not None and inequality.h is not None  # mypy.
            lower_list.append(inequality.lower)
            upper_list.append(inequality.upper)
    if not lower_list:
        return None, None
    print(len(lower_list))
    lower = np.maximum.reduce(lower_list)
    upper = np.minimum.reduce(upper_list)
    return lower, upper


def build_ik(
    configuration: Configuration,
    tasks: Sequence[Task],
    limits: Sequence[Limit],
    dt: float,
    damping: float = 1e-12,
    prev_sol: np.ndarray | None = None,
):
    P, q = _compute_qp_objective(configuration, tasks, damping)
    lower, upper = _compute_qp_inequalities(configuration, limits, dt)
    return Problem.initialize(configuration, P, q, lower, upper, prev_sol)


def solve_ik(
    configuration: Configuration,
    tasks: Sequence[Task],
    limits: Sequence[Limit],
    dt: float,
    damping: float = 1e-12,
    prev_sol: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a velocity tangent to the current configuration."""
    configuration.check_limits(safety_break=False)
    problem = build_ik(configuration, tasks, limits, dt, damping, prev_sol)
    dq = problem.solve()
    assert dq is not None
    v: np.ndarray = dq / dt
    return v
