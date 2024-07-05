from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import mujoco

from mink.tasks import Task
from mink.configuration import Configuration


@dataclass
class ComTask(Task):

    target_com: Optional[np.ndarray]
    cost: np.ndarray
    gain: float
    lm_damping: float

    @staticmethod
    def initialize(
        cost: float,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ) -> ComTask:
        assert 0 <= gain <= 1

        return ComTask(
            target_com=None,
            cost=np.full((3,), cost),
            gain=gain,
            lm_damping=lm_damping,
        )

    def set_target(self, target_com: np.ndarray) -> None:
        self.target_com = target_com.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        desired_com = configuration.data.subtree_com[0]
        self.set_target(desired_com)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        if self.target_com is None:
            raise ValueError("Target COM is not set.")

        error = configuration.data.subtree_com[1] - self.target_com
        return error

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        J = np.empty((3, configuration.model.nv))
        mujoco.mj_jacSubtreeCom(configuration.model, configuration.data, J, 1)
        return J
