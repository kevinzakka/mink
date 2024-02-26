from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from mink.lie import SE3
from mink.tasks import Task
from mink.configuration import Configuration


@dataclass
class FrameTask(Task):
    """Regulate the pose of a robot frame in the world frame."""

    frame_name: str
    frame_type: str
    cost: np.ndarray
    gain: float
    lm_damping: float
    T_WT: Optional[SE3]

    @staticmethod
    def initialize(
        frame_name: str,
        frame_type: str,
        position_cost: float,
        orientation_cost: float,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ) -> FrameTask:
        """Define a new frame task.

        Args:
            model: Mujoco model.
            frame_name: Name of the object whose frame is to be regulated. Must be a
                body, geom, or site in the model.
            frame_type: Type of the object whose frame is to be regulated. Must be
                "body", "geom", or "site".
            position_cost: Contribution of position errors to the normalized cost.
            orientation_cost: Contribution of orientation errors to the normalized cost.
            gain: Task gain in the [0, 1] range.
        """
        assert 0 <= gain <= 1

        cost = np.zeros(6)
        cost[:3] = position_cost
        cost[3:] = orientation_cost

        return FrameTask(
            frame_name=frame_name,
            frame_type=frame_type,
            cost=cost,
            gain=gain,
            lm_damping=lm_damping,
            T_WT=None,
        )

    def set_target(self, T_WT: SE3) -> None:
        self.T_WT = T_WT

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        self.set_target(
            configuration.get_transform_frame_to_world(
                frame_name=self.frame_name,
                frame_type=self.frame_type,
            )
        )

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        if self.T_WT is None:
            raise ValueError("Target transform in world frame not set.")

        T_WE = configuration.get_transform_frame_to_world(
            frame_name=self.frame_name,
            frame_type=self.frame_type,
        )
        T_ET = T_WE.inverse() @ self.T_WT
        return T_ET.log()

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        if self.T_WT is None:
            raise ValueError("Target transform in world frame not set.")

        jac = configuration.get_frame_jacobian(
            frame_name=self.frame_name,
            frame_type=self.frame_type,
        )
        T_WE = configuration.get_transform_frame_to_world(
            frame_name=self.frame_name,
            frame_type=self.frame_type,
        )
        return T_WE.inverse().adjoint() @ jac
