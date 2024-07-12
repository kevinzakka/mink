from typing import Optional

import mujoco
import numpy as np

from .exceptions import FrameNotFound, KeyframeNotFound, UnsupportedFrameType
from .lie import SE3, SO3

_SUPPORTED_OBJ_TYPES = {"body", "geom", "site"}

_TYPE_TO_ENUM = {
    "body": mujoco.mjtObj.mjOBJ_BODY,
    "geom": mujoco.mjtObj.mjOBJ_GEOM,
    "site": mujoco.mjtObj.mjOBJ_SITE,
}

_TYPE_TO_JAC_FUNCTION = {
    "body": mujoco.mj_jacBody,
    "geom": mujoco.mj_jacGeom,
    "site": mujoco.mj_jacSite,
}
_TYPE_TO_POS_ATTRIBUTE = {
    "body": "xpos",
    "geom": "geom_xpos",
    "site": "site_xpos",
}
_TYPE_TO_XMAT_ATTRIBUTE = {
    "body": "xmat",
    "geom": "geom_xmat",
    "site": "site_xmat",
}


class Configuration:
    """A struct that provides convenient access to kinematic quantities such as frame
    transforms and frame jacobians.

    The `update` function ensures the proper forward kinematics functions have been
    called, namely:

    - mujoco.mj_kinematics(model, data)
    - mujoco.mj_comPos(model, data)
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        q: np.ndarray | None = None,
    ):
        """Constructor.

        Args:
            model: An instance of MjModel.
            q: Optional configuration to initialize from. If None, the configuration
            is initialized to the reference configuration `qpos0`.
        """
        self.model = model
        self.data = mujoco.MjData(model)
        self.update(q=q)

    def update_from_keyframe(self, key: str) -> None:
        """Update the configuration from a keyframe and run forward kinematics.

        Args:
            key: The name of the keyframe.

        Raises:
            ValueError: if no key named `key` was found in the model.
        """
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key)
        if key_id == -1:
            raise KeyframeNotFound(key, self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self.update()

    def update(self, q: Optional[np.ndarray] = None) -> None:
        """Run forward kinematics.

        The minimal function call required to get updated frame transforms (aka forward
        kinematics) is `mj_kinematics`. An extra call to `mj_comPos` is needed for
        updated Jacobians.
        """
        if q is not None:
            self.data.qpos = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def get_frame_jacobian(self, frame_name: str, frame_type: str) -> np.ndarray:
        """Compute the Jacobian matrix of a frame velocity.

        Denoting our frame by B and the world frame by W, the Jacobian matrix B_J_WB
        is related to the body velocity B_v_WB by:

            B_v_WB = B_J_WB q_dot

        Args:
            frame_name: Name of the frame in the MJCF.
            frame_type: Type of frame. Can be a geom, a body or a site.

        Returns:
            Jacobian B_J_WB of the frame.
        """
        if frame_type not in _SUPPORTED_OBJ_TYPES:
            raise UnsupportedFrameType(frame_type, _SUPPORTED_OBJ_TYPES)

        frame_id = mujoco.mj_name2id(self.model, _TYPE_TO_ENUM[frame_type], frame_name)
        if frame_id == -1:
            raise FrameNotFound(
                frame_name=frame_name,
                frame_type=frame_type,
                model=self.model,
            )

        jac = np.zeros((6, self.model.nv))
        _TYPE_TO_JAC_FUNCTION[frame_type](
            self.model, self.data, jac[:3], jac[3:], frame_id
        )
        R_sb = getattr(self.data, _TYPE_TO_XMAT_ATTRIBUTE[frame_type])[frame_id]
        R_sb = R_sb.reshape(3, 3).T
        jac[3:] = R_sb @ jac[3:]
        jac[:3] = R_sb @ jac[:3]
        return jac

    def get_transform_frame_to_world(self, frame_name: str, frame_type: str) -> SE3:
        """Get the pose of a frame in the current configuration.

        Denoting our frame by B and the world frame by W, this function returns T_WB.

        Args:
            frame_name: Name of the frame in the MJCF.
            frame_type: Type of frame. Can be a geom, a body or a site.

        Returns:
            The pose of the frame in the world frame.
        """
        if frame_type not in _SUPPORTED_OBJ_TYPES:
            raise UnsupportedFrameType(frame_type, _SUPPORTED_OBJ_TYPES)

        frame_id = mujoco.mj_name2id(self.model, _TYPE_TO_ENUM[frame_type], frame_name)
        if frame_id == -1:
            raise FrameNotFound(
                frame_name=frame_name,
                frame_type=frame_type,
                model=self.model,
            )

        xpos = getattr(self.data, _TYPE_TO_POS_ATTRIBUTE[frame_type])[frame_id]
        xmat = getattr(self.data, _TYPE_TO_XMAT_ATTRIBUTE[frame_type])[frame_id]
        return SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(xmat.reshape(3, 3)),
            translation=xpos,
        )

    def integrate(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        """Integrate a velocity starting from the current configuration.

        Args:
            velocity: The velocity.
            dt: Integration duration in [s].

        Returns:
            The new configuration after integration.
        """
        q = self.data.qpos.copy()
        mujoco.mj_integratePos(self.model, q, velocity, dt)
        return q

    def integrate_inplace(self, velocity: np.ndarray, dt: float) -> None:
        """Integrate a velocity and update the current configuration inplace.

        Args:
            velocity: The velocity.
            dt: Integration duration in [s].
        """
        mujoco.mj_integratePos(self.model, self.data.qpos, velocity, dt)
        self.update()

    @property
    def q(self) -> np.ndarray:
        """The current configuration vector."""
        return self.data.qpos.copy()
