from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from mink.tasks import Task
from typing import Optional
from mink.configuration import Configuration
import mujoco


@dataclass
class PostureTask(Task):
    """Regulate joint angles to a desired posture."""

    cost: float
    gain: float = 1.0
    lm_damping: float = 0.0
    target_q: Optional[np.ndarray] = None

    @staticmethod
    def initialize(
        cost: float,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ) -> PostureTask:
        return PostureTask(
            cost=float(cost),
            gain=float(gain),
            lm_damping=float(lm_damping),
        )

    def set_target(self, target_q: np.ndarray) -> None:
        self.target_q = target_q.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        self.set_target(configuration.q)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        if self.target_q is None:
            raise ValueError("Target joint angles not set.")

        qvel = np.zeros(configuration.model.nv)
        mujoco.mj_differentiatePos(
            m=configuration.model,
            qvel=qvel,
            dt=1.0,
            qpos1=configuration.q,
            qpos2=self.target_q,
        )
        return qvel

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        return -np.eye(configuration.model.nv)
