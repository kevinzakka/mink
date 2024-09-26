"""Build and solve the inverse kinematics problem."""

from typing import Optional, Sequence

from dataclasses import dataclass
import numpy as np
import mujoco

from .configuration import Configuration
from .limits import ConfigurationLimit, Limit
from .tasks import Objective, Task


@dataclass(frozen=True)
class Problem:
    """Wrapper over `mujoco.mju_boxQP`."""

    H: np.ndarray
    c: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    n: int
    R: np.ndarray
    index: np.ndarray

    @staticmethod
    def initialize(
        configuration: Configuration,
        H: np.ndarray,
        c: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ):
        n = configuration.nv
        R = np.zeros((n, n + 7))
        index = np.zeros(n, np.int32)
        return Problem(H, c, lower, upper, n, R, index)

    def solve(self, prev_sol: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if prev_sol is not None:
            dq = prev_sol
            assert dq.shape == (self.n,)
        else:
            dq = np.empty((self.n,))
        rank = mujoco.mju_boxQP(
            res=dq,
            R=self.R,
            index=self.index,
            H=self.H,
            g=self.c,
            lower=self.lower,
            upper=self.upper,
        )
        if rank == -1:
            return None
        return dq


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
    lower_list: list[np.ndarray] = []
    upper_list: list[np.ndarray] = []
    for limit in limits:
        inequality = limit.compute_qp_inequalities(configuration, dt)
        if not inequality.inactive:
            assert inequality.lower is not None and inequality.lower is not None  # mypy.
            lower_list.append(inequality.lower)
            upper_list.append(inequality.upper)
    if not lower_list:
        return None, None
    lower = np.maximum.reduce(lower_list)
    upper = np.minimum.reduce(upper_list)
    return lower, upper


def build_ik(
    configuration: Configuration,
    tasks: Sequence[Task],
    dt: float,
    damping: float = 1e-12,
    limits: Optional[Sequence[Limit]] = None,
) -> Problem:
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
    lower, upper = _compute_qp_inequalities(configuration, limits, dt)
    return Problem.initialize(configuration, P, q, lower, upper)


def solve_ik(
    configuration: Configuration,
    tasks: Sequence[Task],
    dt: float,
    damping: float = 1e-12,
    safety_break: bool = False,
    limits: Optional[Sequence[Limit]] = None,
    prev_sol: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve the differential inverse kinematics problem.

    Computes a velocity tangent to the current robot configuration. The computed
    velocity satisfies at (weighted) best the set of provided kinematic tasks.

    Args:
        configuration: Robot configuration.
        tasks: List of kinematic tasks.
        dt: Integration timestep in [s].
        damping: Levenberg-Marquardt damping.
        safety_break: If True, stop execution and raise an exception if
            the current configuration is outside limits. If False, print a
            warning and continue execution.
        limits: List of limits to enforce. Set to empty list to disable. If None,
            defaults to a configuration limit.

    Returns:
        Velocity `v` in tangent space.
    """
    configuration.check_limits(safety_break=safety_break)
    problem = build_ik(configuration, tasks, dt, damping, limits)
    dq = problem.solve(prev_sol)
    assert dq is not None
    v: np.ndarray = dq / dt
    return v
