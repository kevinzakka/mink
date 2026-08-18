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
        r"""Compute the relative frame task error.

        The error is the right-minus difference between the target pose
        :math:`T_{rf}^*` and the current pose :math:`T_{rf}` of our frame,
        both relative to the root frame :math:`r`:

        .. math::

            e(q) = T_{rf}^* \ominus T_{rf}
            = \log(T_{rf}^{-1} \cdot T_{rf}^*) = \log E_{rf}

        The root is where the poses are compared; the resulting twist is
        expressed in the controlled frame :math:`f`, matching the
        :class:`~mink.tasks.frame_task.FrameTask` convention with the world
        frame replaced by the root frame.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Relative frame task error vector :math:`e(q)`.
        """
        if self.transform_target_to_root is None:
            raise TargetNotSet(self.__class__.__name__)

        frame = configuration._get_transform_wxyz_xyz(
            self.frame_name, self.frame_type, self.root_name, self.root_type
        )
        target = self.transform_target_to_root.wxyz_xyz
        if _native is not None:
            return _native.se3_rminus(target, frame)
        return SE3(wxyz_xyz=target).rminus(SE3(wxyz_xyz=frame))

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the relative frame task Jacobian.

        The Jacobian is the derivative of the task error with respect to the
        configuration displacement :math:`\Delta q`:

        .. math::

            J(q) = -\text{Jlog}_6(E_{rf}^{-1}) \,
            \left({}_f J_{0f} - \text{Ad}(T_{fr}) \, {}_r J_{0r}\right)

        where :math:`E_{rf} = T_{rf}^{-1} T_{rf}^*` is the pose residual,
        :math:`T_{fr} = T_{rf}^{-1}`, and :math:`{}_f J_{0f}`,
        :math:`{}_r J_{0r}` are the body Jacobians of the frame and the root.
        See :ref:`derivations` for how this formula is derived and evaluated
        from MuJoCo quantities.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Relative frame task jacobian :math:`J(q)`.
        """
        if self.transform_target_to_root is None:
            raise TargetNotSet(self.__class__.__name__)

        target = self.transform_target_to_root.wxyz_xyz

        if _native is not None:
            frame_world = configuration._get_transform_frame_to_world_wxyz_xyz(
                self.frame_name, self.frame_type
            )
            root_world = configuration._get_transform_frame_to_world_wxyz_xyz(
                self.root_name, self.root_type
            )
            jac_frame = configuration._get_frame_jacobian_world_aligned(
                self.frame_name, self.frame_type
            )
            jac_root = configuration._get_frame_jacobian_world_aligned(
                self.root_name, self.root_type
            )
            _, M_frame, M_root = _native.se3_relative_frame_task_terms(
                target, frame_world, root_world
            )
            return M_frame @ jac_frame + M_root @ jac_root

        jacobian_frame_in_frame = configuration.get_frame_jacobian(
            self.frame_name, self.frame_type
        )
        jacobian_root_in_root = configuration.get_frame_jacobian(
            self.root_name, self.root_type
        )
        frame = configuration._get_transform_wxyz_xyz(
            self.frame_name, self.frame_type, self.root_name, self.root_type
        )
        frame_se3 = SE3(wxyz_xyz=frame)
        target_se3 = SE3(wxyz_xyz=target)
        E_inv = target_se3.inverse() @ frame_se3
        return -E_inv.jlog() @ (
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

        target = self.transform_target_to_root.wxyz_xyz

        if _native is not None:
            frame_world = configuration._get_transform_frame_to_world_wxyz_xyz(
                self.frame_name, self.frame_type
            )
            root_world = configuration._get_transform_frame_to_world_wxyz_xyz(
                self.root_name, self.root_type
            )
            jac_frame = configuration._get_frame_jacobian_world_aligned(
                self.frame_name, self.frame_type
            )
            jac_root = configuration._get_frame_jacobian_world_aligned(
                self.root_name, self.root_type
            )
            error, M_frame, M_root = _native.se3_relative_frame_task_terms(
                target, frame_world, root_world
            )
            jacobian = M_frame @ jac_frame + M_root @ jac_root
        else:
            jacobian_frame_in_frame = configuration.get_frame_jacobian(
                self.frame_name, self.frame_type
            )
            jacobian_root_in_root = configuration.get_frame_jacobian(
                self.root_name, self.root_type
            )
            frame = configuration._get_transform_wxyz_xyz(
                self.frame_name, self.frame_type, self.root_name, self.root_type
            )
            frame_se3 = SE3(wxyz_xyz=frame)
            target_se3 = SE3(wxyz_xyz=target)
            error = target_se3.rminus(frame_se3)
            E_inv = target_se3.inverse() @ frame_se3
            jacobian = -E_inv.jlog() @ (
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
