"""Damping task implementation."""

from __future__ import annotations

import mujoco
import numpy.typing as npt

from .posture_task import PostureTask


class DampingTask(PostureTask):
    """Minimize joint velocities.

    This task is implemented as a special case of the PostureTask where the gain and
    target configuration are set to 0 and qpos0 respectively.
    """

    def __init__(self, model: mujoco.MjModel, cost: npt.ArrayLike):
        super().__init__(model=model, cost=cost, gain=0.0, lm_damping=0.0)
        self.target_q = model.qpos0
