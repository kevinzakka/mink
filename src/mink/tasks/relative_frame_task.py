"""Relative frame task implementation."""

from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..exceptions import TargetNotSet, TaskDefinitionError
from ..lie import SE3
from .task import Objective, Task

try:
    if os.environ.get("MINK_DISABLE_NATIVE", ""):
        raise ImportError
    from ..lie import _lie_ops_c as _native  # noqa: PLC0415
except ImportError:
    _native = None  # type: ignore[assignment]


class RelativeFrameTask(Task):
    """Regulate the pose of a frame relative to another frame.

    Attributes:
        frame_name: Name or id of the frame to regulate, typically a body, geom
            or site in the robot model.
        frame_type: The frame type: `body`, `geom` or `site`.
        root_name: Name or id of the frame the task is relative to.
        root_type: The root frame type: `body`, `geom` or `site`.
        transform_target_to_root: Target pose in the root frame.
    """

    k: int = 6
    transform_target_to_root: SE3 | None

    def __init__(
        self,
        frame_name: str | int,
        frame_type: str,
        root_name: str | int,
        root_type: str,
        position_cost: npt.ArrayLike,
        orientation_cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        super().__init__(cost=np.zeros((self.k,)), gain=gain, lm_damping=lm_damping)
        self.frame_name = frame_name
        self.frame_type = frame_type
        self.root_name = root_name
        self.root_type = root_type
        self.position_cost = position_cost
        self.orientation_cost = orientation_cost
        self.transform_target_to_root = None

        self.set_position_cost(position_cost)
        self.set_orientation_cost(orientation_cost)

    def set_position_cost(self, position_cost: npt.ArrayLike) -> None:
        position_cost = np.atleast_1d(position_cost)
        if position_cost.ndim != 1 or position_cost.shape[0] not in (1, 3):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} position cost should be a vector of shape "
                "1 (aka identical cost for all coordinates) or (3,) but got "
                f"{position_cost.shape}"
            )
        if not np.all(position_cost >= 0.0):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} position cost should be >= 0"
            )
        self.cost[:3] = position_cost

    def set_orientation_cost(self, orientation_cost: npt.ArrayLike) -> None:
        orientation_cost = np.atleast_1d(orientation_cost)
        if orientation_cost.ndim != 1 or orientation_cost.shape[0] not in (1, 3):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} orientation cost should be a vector of "
                "shape 1 (aka identical cost for all coordinates) or (3,) but got "
                f"{orientation_cost.shape}"
            )
        if not np.all(orientation_cost >= 0.0):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} position cost should be >= 0"
            )
        self.cost[3:] = orientation_cost

    def set_target(self, transform_target_to_root: SE3) -> None:
        """Set the target pose in the root frame.

        Args:
            transform_target_to_root: Transform from the task target frame to the
                root frame.
        """
        self.transform_target_to_root = transform_target_to_root.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set the target pose from a given robot configuration.

        Args:
            configuration: Robot configuration :math:`q`.
        """
        self.set_target(
            configuration.get_transform(
                self.frame_name,
                self.frame_type,
                self.root_name,
                self.root_type,
            )
        )

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        if self.transform_target_to_root is None:
            raise TargetNotSet(self.__class__.__name__)

        frame = configuration._get_transform_wxyz_xyz(
            self.frame_name, self.frame_type, self.root_name, self.root_type
        )
        target = self.transform_target_to_root.wxyz_xyz
        if _native is not None:
            return _native.se3_rminus(frame, target)
        return SE3(wxyz_xyz=frame).rminus(SE3(wxyz_xyz=target))

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        if self.transform_target_to_root is None:
            raise TargetNotSet(self.__class__.__name__)

        jacobian_frame_in_frame = configuration.get_frame_jacobian(
            self.frame_name, self.frame_type
        )
        jacobian_root_in_root = configuration.get_frame_jacobian(
            self.root_name, self.root_type
        )
        frame = configuration._get_transform_wxyz_xyz(
            self.frame_name, self.frame_type, self.root_name, self.root_type
        )
        target = self.transform_target_to_root.wxyz_xyz

        if _native is not None:
            T_ft = _native.se3_inverse_multiply(target, frame)
            A_rf = _native.se3_adjoint(_native.se3_inverse(frame))
            return _native.se3_jlog(T_ft) @ (
                jacobian_frame_in_frame - A_rf @ jacobian_root_in_root
            )
        else:
            frame_se3 = SE3(wxyz_xyz=frame)
            target_se3 = SE3(wxyz_xyz=target)
            transform_frame_to_target = target_se3.inverse() @ frame_se3
            return transform_frame_to_target.jlog() @ (
                jacobian_frame_in_frame
                - frame_se3.inverse().adjoint() @ jacobian_root_in_root
            )

    def _error_and_jacobian(
        self, configuration: Configuration
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the task error and Jacobian together, sharing the relative
        transform and frame Jacobians that both depend on."""
        if self.transform_target_to_root is None:
            raise TargetNotSet(self.__class__.__name__)

        jacobian_frame_in_frame = configuration.get_frame_jacobian(
            self.frame_name, self.frame_type
        )
        jacobian_root_in_root = configuration.get_frame_jacobian(
            self.root_name, self.root_type
        )
        frame = configuration._get_transform_wxyz_xyz(
            self.frame_name, self.frame_type, self.root_name, self.root_type
        )
        target = self.transform_target_to_root.wxyz_xyz

        if _native is not None:
            error = _native.se3_rminus(frame, target)
            T_ft = _native.se3_inverse_multiply(target, frame)
            A_rf = _native.se3_adjoint(_native.se3_inverse(frame))
            jacobian = _native.se3_jlog(T_ft) @ (
                jacobian_frame_in_frame - A_rf @ jacobian_root_in_root
            )
        else:
            frame_se3 = SE3(wxyz_xyz=frame)
            target_se3 = SE3(wxyz_xyz=target)
            error = frame_se3.rminus(target_se3)
            transform_frame_to_target = target_se3.inverse() @ frame_se3
            jacobian = transform_frame_to_target.jlog() @ (
                jacobian_frame_in_frame
                - frame_se3.inverse().adjoint() @ jacobian_root_in_root
            )

        return error, jacobian

    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        r"""Compute the matrix-vector pair :math:`(H, c)` of the QP objective.

        Overrides the base implementation to compute shared quantities once.
        """
        error, jacobian = self._error_and_jacobian(configuration)
        return self._assemble_qp(error, jacobian, configuration._eye_nv)

    def compute_qp_residual(
        self, configuration: Configuration
    ) -> tuple[np.ndarray, np.ndarray, float]:
        error, jacobian = self._error_and_jacobian(configuration)
        return self._weighted_residual(error, jacobian)
