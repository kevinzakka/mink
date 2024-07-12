import mujoco
import numpy as np
import pinocchio as pin

from mink.configuration import Configuration
from mink.lie import SE3, SO3
from mink.tasks import Task


class FrameTask(Task):
    """Regulate the pose of a robot frame in the world frame."""

    def __init__(
        self,
        frame_name: str,
        frame_type: str,
        position_cost: float,
        orientation_cost: float,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        cost = np.zeros((6,), dtype=np.float64)
        cost[:3] = position_cost
        cost[3:] = orientation_cost

        super().__init__(cost=cost, gain=gain, lm_damping=lm_damping)

        self.frame_name = frame_name
        self.frame_type = frame_type
        self.world_T_target = None

    def set_target_from_mocap(self, data: mujoco.MjData, mocap_id: int) -> None:
        self.set_target(
            SE3.from_rotation_and_translation(
                rotation=SO3(data.mocap_quat[mocap_id]),
                translation=data.mocap_pos[mocap_id],
            )
        )

    def set_target(self, world_T_target: SE3) -> None:
        self.world_T_target = world_T_target

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        self.set_target(
            configuration.get_transform_frame_to_world(
                frame_name=self.frame_name,
                frame_type=self.frame_type,
            )
        )

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        if self.world_T_target is None:
            raise ValueError("Target transform in world frame not set.")
        T_0b = configuration.get_transform_frame_to_world(
            frame_name=self.frame_name,
            frame_type=self.frame_type,
        )
        T_0t = self.world_T_target
        T_bt = T_0b.inverse() @ T_0t
        return T_bt.log()

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        if self.world_T_target is None:
            raise ValueError("Target transform in world frame not set.")
        T_0b = configuration.get_transform_frame_to_world(
            frame_name=self.frame_name,
            frame_type=self.frame_type,
        )
        T_0t = self.world_T_target
        jac = configuration.get_frame_jacobian(
            frame_name=self.frame_name,
            frame_type=self.frame_type,
        )
        T_tb = T_0t.inverse() @ T_0b
        return -pin.Jlog6(pin.SE3(T_tb.as_matrix())) @ jac
