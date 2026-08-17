"""Frame task implementation."""

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


class FrameTask(Task):
    """Regulate the position and orientation of a frame of interest on the robot.

    Attributes:
        frame_name: Name or id of the frame to regulate, typically a body, geom
            or site in the robot model.
        frame_type: The frame type: `body`, `geom` or `site`.
        transform_target_to_world: Target pose of the frame in the world frame.

    Example:

    .. code-block:: python

        frame_task = FrameTask(
            frame_name="target",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
        )

        # Update the target pose directly.
        transform_target_to_world = SE3.from_translation(np.random.rand(3))
        frame_task.set_target(transform_target_to_world)

        # Or from the current configuration. This will automatically compute the
        # target pose from the current configuration and update the task target.
        frame_task.set_target_from_configuration(configuration)
    """

    k: int = 6
    transform_target_to_world: SE3 | None

    def __init__(
        self,
        frame_name: str | int,
        frame_type: str,
        position_cost: npt.ArrayLike,
        orientation_cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        super().__init__(cost=np.zeros((self.k,)), gain=gain, lm_damping=lm_damping)
        self.frame_name = frame_name
        self.frame_type = frame_type
        self.position_cost = position_cost
        self.orientation_cost = orientation_cost
        self.transform_target_to_world = None

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

    def set_target(self, transform_target_to_world: SE3) -> None:
        """Set the target pose.

        Args:
            transform_target_to_world: Transform from the task target frame to the
                world frame.
        """
        self.transform_target_to_world = transform_target_to_world.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set the target pose from a given robot configuration.

        Args:
            configuration: Robot configuration :math:`q`.
        """
        self.set_target(
            configuration.get_transform_frame_to_world(self.frame_name, self.frame_type)
        )

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the frame task error.

        This error is a twist :math:`e(q) \in se(3)` expressed in the local frame,
        i.e., it is a body twist. It is computed by taking the right-minus difference
        between the target pose :math:`T_{0t}` and current frame pose :math:`T_{0b}`:

        .. math::

            e(q) := {}_b \xi_{0b} = -(T_{t0} \ominus T_{b0})
            = -\log(T_{t0} \cdot T_{0b}) = -\log(T_{tb}) = \log(T_{bt})

        where :math:`b` denotes our frame, :math:`t` the target frame and
        :math:`0` the inertial frame.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Frame task error vector :math:`e(q)`.
        """
        if self.transform_target_to_world is None:
            raise TargetNotSet(self.__class__.__name__)

        frame = configuration._get_transform_frame_to_world_wxyz_xyz(
            self.frame_name, self.frame_type
        )
        if _native is not None:
            return _native.se3_rminus(self.transform_target_to_world.wxyz_xyz, frame)
        target = SE3(wxyz_xyz=self.transform_target_to_world.wxyz_xyz)
        return target.minus(SE3(wxyz_xyz=frame))

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the frame task Jacobian.

        The derivation of the formula for this Jacobian is detailed in
        [FrameTaskJacobian]_.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Frame task jacobian :math:`J(q)`.
        """
        if self.transform_target_to_world is None:
            raise TargetNotSet(self.__class__.__name__)

        jac = configuration.get_frame_jacobian(self.frame_name, self.frame_type)
        frame = configuration._get_transform_frame_to_world_wxyz_xyz(
            self.frame_name, self.frame_type
        )
        target = self.transform_target_to_world.wxyz_xyz
        if _native is not None:
            T_tb = _native.se3_inverse_multiply(target, frame)
            return -_native.se3_jlog(T_tb) @ jac
        T_tb = SE3(wxyz_xyz=target).inverse() @ SE3(wxyz_xyz=frame)
        return -T_tb.jlog() @ jac

    def _error_and_jacobian(
        self, configuration: Configuration
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the task error and Jacobian together, sharing the frame
        transform (which both depend on) so it is only fetched once."""
        if self.transform_target_to_world is None:
            raise TargetNotSet(self.__class__.__name__)

        frame = configuration._get_transform_frame_to_world_wxyz_xyz(
            self.frame_name, self.frame_type
        )
        jac = configuration.get_frame_jacobian(self.frame_name, self.frame_type)
        target = self.transform_target_to_world.wxyz_xyz

        if _native is not None:
            error = _native.se3_rminus(target, frame)
            T_tb = _native.se3_inverse_multiply(target, frame)
            jacobian = -_native.se3_jlog(T_tb) @ jac
        else:
            target_se3 = SE3(wxyz_xyz=target)
            frame_se3 = SE3(wxyz_xyz=frame)
            error = target_se3.minus(frame_se3)
            T_tb = target_se3.inverse() @ frame_se3
            jacobian = -T_tb.jlog() @ jac

        return error, jacobian

    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        r"""Compute the matrix-vector pair :math:`(H, c)` of the QP objective.

        Overrides the base implementation to compute the frame transform once
        and reuse it for both error and Jacobian computation.
        """
        error, jacobian = self._error_and_jacobian(configuration)
        return self._assemble_qp(error, jacobian, configuration._eye_nv)

    def compute_qp_residual(
        self, configuration: Configuration
    ) -> tuple[np.ndarray, np.ndarray, float]:
        error, jacobian = self._error_and_jacobian(configuration)
        return self._weighted_residual(error, jacobian)
