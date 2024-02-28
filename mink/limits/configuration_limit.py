from dataclasses import dataclass
from typing import Sequence
import numpy as np
import mujoco

from mink.limits import Limit, BoxConstraint

_SUPPORTED_JOINT_TYPES = {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}
_INVALID_JOINT_ERROR = "Joint with name {} does not exist."
_UNSUPPORTED_JOINT_ERROR = (
    "Only 1 DoF joints (hinge and slider) are supported at the moment. Joint with name "
    "{} is not a 1 DoF joint (type {})."
)


def _get_joint_ids(
    model: mujoco.MjModel,
    joints: Sequence[str],
) -> np.ndarray:
    """Get the joint IDs from a sequence of joint names.

    Joints must be 1 DoF, i.e., hinge or slider joints.
    """
    joint_ids: list[int] = []
    for joint in joints:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if joint_id == -1:
            raise ValueError(_INVALID_JOINT_ERROR.format(joint))
        joint_type = model.jnt_type[joint_id]
        if joint_type not in _SUPPORTED_JOINT_TYPES:
            raise ValueError(_UNSUPPORTED_JOINT_ERROR.format(joint, joint_type))
        joint_ids.append(joint_id)
    return np.array(joint_ids)


@dataclass(frozen=True)
class ConfigurationLimit(Limit):
    """A subspace of joint velocities restricted to joints with position limits.

    Currently only 1 DoF joints are supported, aka hinge and slider joints.

    Attributes:
        model: MuJoCo model.
        indices: Indices of joints with configuration limits.
        projection_matrix: Projection from tangent space to subspace with
            configuration-limited joints.
        upper_limits: Upper position limits for each joint.
        lower_limits: Lower position limits for each joint.
        limit_gain: Gain factor between 0 and 1 that defines how fast each joint is
            allowed to move towards its limit in each integration step.
    """

    model: mujoco.MjModel
    indices: np.ndarray
    projection_matrix: np.ndarray
    upper_limits: np.ndarray
    lower_limits: np.ndarray
    limit_gain: float

    @staticmethod
    def initialize(
        model: mujoco.MjModel,
        joints: Sequence[str],
        limit_gain: float = 0.5,
    ) -> "ConfigurationLimit":
        """Initialize configuration limits.

        Configuration limits are automatically extracted from the model.

        Args:
            model: MuJoCo model.
            joints: Sequence of joint names to be configuration-limited. Joints must be
                1 DoF, i.e., hinge or slider joints.
            limit_gain: Gain factor between 0 and 1 that determines the percentage of
                maximum velocity allowed in each timestep.
        """
        if not 0.0 < limit_gain < 1.0:
            raise ValueError("Limit gain must be in the range (0, 1).")

        indices = _get_joint_ids(model, joints)
        for idx in indices:
            is_limited = model.jnt_limited[idx]
            if not is_limited:
                jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, idx)
                raise ValueError(
                    f"Joint {jnt_name} does not have limits. This constraint "
                    "can only be defined for 1 DoF joints with limits."
                )

        return ConfigurationLimit(
            model=model,
            indices=indices,
            projection_matrix=np.eye(model.nv)[indices],
            upper_limits=model.jnt_range[:, 0],
            lower_limits=model.jnt_range[:, 1],
            limit_gain=limit_gain,
        )

    def compute_qp_inequalities(self, q: np.ndarray, dt: float) -> BoxConstraint:
        del dt  # Unused.

        delta_q_max = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(
            m=self.model,
            qvel=delta_q_max,
            dt=1.0,
            qpos1=self.upper_limits,
            qpos2=q,
        )
        upper = self.limit_gain * delta_q_max[self.indices]

        delta_q_min = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(
            m=self.model,
            qvel=delta_q_min,
            dt=1.0,
            qpos1=self.lower_limits,
            qpos2=q,
        )
        lower = self.limit_gain * delta_q_min[self.indices]

        return BoxConstraint(lower=lower, upper=upper)
