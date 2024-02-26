from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import mujoco
from mink.lie import SE3, SO3

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


@dataclass(frozen=True)
class Configuration:
    model: mujoco.MjModel
    data: mujoco.MjData
    q: np.ndarray

    @staticmethod
    def initialize(
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> Configuration:
        return Configuration(model, data, data.qpos)

    @staticmethod
    def initialize_from_keyframe(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        keyframe_name: str,
    ) -> Configuration:
        keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
        if not 0 <= keyframe_id < model.nkey:
            raise ValueError(f"Keyframe '{keyframe_name}' not found")
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
        return Configuration(model, data, data.qpos)

    def update(self) -> None:
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def get_frame_jacobian(self, frame_name: str, frame_type: str) -> np.ndarray:
        assert frame_type in _SUPPORTED_OBJ_TYPES
        frame_id = mujoco.mj_name2id(self.model, _TYPE_TO_ENUM[frame_type], frame_name)
        if frame_id == -1:
            raise ValueError(f"Frame '{frame_name}' not found")
        jac = np.zeros((6, self.model.nv))
        _TYPE_TO_JAC_FUNCTION[frame_type](
            self.model, self.data, jac[:3], jac[3:], frame_id
        )
        return jac

    def get_transform_frame_to_world(self, frame_name: str, frame_type: str) -> SE3:
        assert frame_type in _SUPPORTED_OBJ_TYPES
        frame_id = mujoco.mj_name2id(self.model, _TYPE_TO_ENUM[frame_type], frame_name)
        if frame_id == -1:
            raise ValueError(f"Frame '{frame_name}' not found")
        xpos = getattr(self.data, _TYPE_TO_POS_ATTRIBUTE[frame_type])[frame_id]
        xmat = getattr(self.data, _TYPE_TO_XMAT_ATTRIBUTE[frame_type])[frame_id]
        return SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(xmat.reshape(3, 3)),
            translation=xpos,
        )

    def integrate(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        q = self.q.copy()
        mujoco.mj_integratePos(self.model, q, velocity, dt)
        return q

    def integrate_in_place(self, velocity: np.ndarray, dt: float) -> None:
        mujoco.mj_integratePos(self.model, self.q, velocity, dt)
        self.update()
