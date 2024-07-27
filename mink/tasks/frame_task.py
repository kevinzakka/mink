"""Frame task implementation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..lie import SE3
from .exceptions import TargetNotSet, TaskDefinitionError
from .task import Task


class FrameTask(Task):
    """Regulate the pose of a frame expressed in the world frame.

    Attributes:
        frame_name: Name of the frame to regulate, typically the name of body, geom
            or site in the robot model.
        frame_type: The frame type: `body`, `geom` or `site`.
        transform_frame_to_world: Target pose of the frame.
    """

    k: int = 6
    transform_target_to_world: Optional[SE3]

    def __init__(
        self,
        frame_name: str,
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

        transform_frame_to_world = configuration.get_transform_frame_to_world(
            self.frame_name, self.frame_type
        )
        return self.transform_target_to_world.minus(transform_frame_to_world)

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

        transform_frame_to_world = configuration.get_transform_frame_to_world(
            self.frame_name, self.frame_type
        )

        T_tb = self.transform_target_to_world.inverse() @ transform_frame_to_world
        return -T_tb.jlog() @ jac
