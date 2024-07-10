import mujoco
import numpy as np

from mink.lie import SE3, SO3
from mink.tasks import Task
from mink.configuration import Configuration


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

        ## OLD WAY OF DOING IT
        # world_T_frame = configuration.get_transform_frame_to_world(
        #     frame_name=self.frame_name,
        #     frame_type=self.frame_type,
        # )
        # frame_T_target = world_T_frame.inverse() @ self.world_T_target
        # return frame_T_target.log()

        world_T_frame = configuration.get_transform_frame_to_world(
            frame_name=self.frame_name,
            frame_type=self.frame_type,
        )
        error = np.zeros(6)
        error[:3] = world_T_frame.translation() - self.world_T_target.translation()
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, world_T_frame.rotation().as_matrix().ravel())
        mujoco.mju_subQuat(error[3:], self.world_T_target.rotation().wxyz, site_quat)

        return error

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        if self.world_T_target is None:
            raise ValueError("Target transform in world frame not set.")

        jac = configuration.get_frame_jacobian(
            frame_name=self.frame_name,
            frame_type=self.frame_type,
        )
        world_T_frame = configuration.get_transform_frame_to_world(
            frame_name=self.frame_name,
            frame_type=self.frame_type,
        )

        ## OLD WAY OF DOING IT
        # return -world_T_frame.inverse().adjoint() @ jac

        effector_quat = np.empty(4)
        mujoco.mju_mat2Quat(effector_quat, world_T_frame.rotation().as_matrix().ravel())
        target_quat = self.world_T_target.rotation().wxyz
        Deffector = np.empty((3, 3))
        mujoco.mjd_subQuat(target_quat, effector_quat, None, Deffector)
        target_mat = self.world_T_target.rotation().as_matrix()
        mat = Deffector.T @ target_mat.T
        jac[3:] = mat @ jac[3:]

        return jac
