"""Exceptions specific to mink."""

import mujoco


class MinkError(Exception):
    """Base class for Mink exceptions."""


class UnsupportedFrameType(MinkError):
    """Exception raised when a frame type is unsupported."""

    def __init__(self, frame_type: str, supported_types: list[str]):
        message = (
            f"{frame_type} is not supported."
            f"Supported frame types are: {supported_types}"
        )
        super().__init__(message)


class FrameNotFound(MinkError):
    """Exception raised when a frame is not found in the robot model."""

    def __init__(
        self,
        frame_name: str,
        frame_type: str,
        model: mujoco.MjModel,
    ):
        if frame_type == "body":
            available_names_of_type_frame_type = [
                model.body(i).name for i in range(model.nbody)
            ]
        elif frame_type == "site":
            available_names_of_type_frame_type = [
                model.site(i).name for i in range(model.nsite)
            ]
        else:
            assert frame_type == "geom"
            available_names_of_type_frame_type = [
                model.geom(i).name for i in range(model.ngeom)
            ]

        message = (
            f"Frame '{frame_name}' does not exist in the model. "
            f"Available {frame_type} names: {available_names_of_type_frame_type}"
        )

        super().__init__(message)


class KeyframeNotFound(MinkError):
    """Exception raised when a keyframe is not found in the robot model."""

    def __init__(self, keyframe_name: str, model: mujoco.MjModel):
        available_keyframes = [model.key(i).name for i in range(model.nkey)]
        message = (
            f"Keyframe {keyframe_name} does not exist in the model. "
            f"Available keyframe names: {available_keyframes}"
        )
        super().__init__(message)
