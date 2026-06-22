"""Axis-align task implementation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..exceptions import InvalidTarget, TargetNotSet, TaskDefinitionError
from ..lie.utils import get_epsilon, skew
from .task import Objective, Task


class AxisAlignTask(Task):
    r"""Align a frame's axis with a target direction in the world.

    This is the sibling of :class:`LookAtTask`: look-at points a frame's axis at a
    *point*, while axis-align points it along a *direction* (a surface normal,
    gravity, a machine axis), i.e. a target at infinity. As with look-at, only the
    pointing of the axis is constrained, not the roll about it: the task error and
    Jacobian both have rank two, leaving rotation about the axis free for other
    tasks to use.

    A frame :math:`B` is given a fixed unit *axis* :math:`a` expressed in its own
    coordinates. Denoting the frame rotation in the world by :math:`R` and the
    target direction by the unit vector :math:`t`, the target expressed in the
    frame is :math:`d = R^T t`, and the task error is the misalignment of the axis
    with that direction:

    .. math::

        e(q) = a - R^T t.

    Differentiating with respect to the configuration, using the angular block
    :math:`J_\omega` of the body Jacobian :math:`{}_B J_{WB} = [J_v; J_\omega]`
    returned by :meth:`Configuration.get_frame_jacobian` (expressed in the local
    frame), gives the task Jacobian

    .. math::

        J(q) = -[R^T t]_\times J_\omega,

    where :math:`[\cdot]_\times` is the skew-symmetric matrix. Because the target
    is a direction rather than a point, only rotation of the frame changes the
    error, so unlike look-at there is no translation term. The skew matrix of a
    non-zero vector has rank two, which is what makes the task rank two.

    The task is degenerate when the axis is anti-aligned with the target
    (:math:`R^T t = -a`); there the Levenberg-Marquardt ``lm_damping`` keeps the
    step bounded.

    Attributes:
        frame_name: Name of the frame to control, typically a body, geom or site.
        frame_type: The frame type: ``body``, ``geom`` or ``site``.
        axis: Unit axis in the frame's local coordinates.
        target_dir: Target direction to align with, in the world frame.

    Example:

    .. code-block:: python

        # Keep a tool's local +z axis aligned with the world up direction.
        axis_align_task = AxisAlignTask(
            frame_name="tool",
            frame_type="site",
            axis=(0.0, 0.0, 1.0),
            cost=1.0,
        )
        axis_align_task.set_target(np.array([0.0, 0.0, 1.0]))
    """

    k: int = 3
    target_dir: np.ndarray | None

    def __init__(
        self,
        frame_name: str,
        frame_type: str,
        axis: npt.ArrayLike = (0.0, 0.0, 1.0),
        cost: npt.ArrayLike = 1.0,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        """Constructor.

        Args:
            frame_name: Name of the frame to control.
            frame_type: The frame type: ``body``, ``geom`` or ``site``.
            axis: Axis in the frame's local coordinates. Need not be normalized; it
                is normalized internally. Defaults to the local ``+z`` axis.
            cost: Cost of the axis-align task. A scalar or a vector of shape (1,).
            gain: Task gain in [0, 1] for additional low-pass filtering.
            lm_damping: Levenberg-Marquardt damping for ill-conditioned targets.
        """
        super().__init__(cost=np.zeros((self.k,)), gain=gain, lm_damping=lm_damping)
        self.frame_name = frame_name
        self.frame_type = frame_type
        self.target_dir = None

        self.set_axis(axis)
        self.set_cost(cost)

    def set_axis(self, axis: npt.ArrayLike) -> None:
        """Set the local axis, normalizing to unit length."""
        axis = np.atleast_1d(np.asarray(axis, dtype=float))
        if axis.shape != (3,):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} axis must have shape (3,) but got "
                f"{axis.shape}"
            )
        norm = np.linalg.norm(axis)
        if norm < get_epsilon(axis.dtype):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} axis must be non-zero"
            )
        self.axis = axis / norm

    def set_cost(self, cost: npt.ArrayLike) -> None:
        """Set the cost of the axis-align task.

        Args:
            cost: A scalar, or a vector of shape (1,). The axis-align error is a
                direction, so a single cost weights all of its coordinates.
        """
        cost = np.atleast_1d(cost)
        if cost.ndim != 1 or cost.shape[0] != 1:
            raise TaskDefinitionError(
                f"{self.__class__.__name__} cost must be a scalar or a vector of "
                f"shape (1,). Got {cost.shape}"
            )
        if not np.all(cost >= 0.0):
            raise TaskDefinitionError(f"{self.__class__.__name__} cost must be >= 0")
        self.cost[:] = cost

    def set_target(self, target_dir: npt.ArrayLike) -> None:
        """Set the target direction to align with, in the world frame.

        Args:
            target_dir: A vector of shape (3,). Need not be normalized; it is
                normalized internally.
        """
        target_dir = np.atleast_1d(np.asarray(target_dir, dtype=float))
        if target_dir.shape != (self.k,):
            raise InvalidTarget(
                f"Expected target direction to have shape ({self.k},) but got "
                f"{target_dir.shape}"
            )
        norm = np.linalg.norm(target_dir)
        if norm < get_epsilon(target_dir.dtype):
            raise InvalidTarget(
                f"{self.__class__.__name__} target direction must be non-zero"
            )
        self.target_dir = target_dir / norm

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set the target to the current world direction of the axis.

        This makes the task a no-op at the given configuration, useful for
        initialization.

        Args:
            configuration: Robot configuration :math:`q`.
        """
        transform = configuration.get_transform_frame_to_world(
            self.frame_name, self.frame_type
        )
        self.set_target(transform.rotation().as_matrix() @ self.axis)

    def _error_and_jacobian(
        self, configuration: Configuration
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.target_dir is None:
            raise TargetNotSet(self.__class__.__name__)

        transform = configuration.get_transform_frame_to_world(
            self.frame_name, self.frame_type
        )
        rotation = transform.rotation().as_matrix()
        jac_w = configuration.get_frame_jacobian(self.frame_name, self.frame_type)[3:]

        # Target direction expressed in the local frame.
        d = rotation.T @ self.target_dir
        error = self.axis - d
        jacobian = -skew(d) @ jac_w
        return error, jacobian

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the axis-align task error.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Axis-align task error :math:`e(q) = a - R^T t`.
        """
        error, _ = self._error_and_jacobian(configuration)
        return error

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the axis-align task Jacobian.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Axis-align task Jacobian :math:`J(q)`.
        """
        _, jacobian = self._error_and_jacobian(configuration)
        return jacobian

    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        r"""Compute the matrix-vector pair :math:`(H, c)` of the QP objective.

        Overrides the base implementation to compute the frame transform and
        Jacobian once and reuse them for both the error and the Jacobian.
        """
        error, jacobian = self._error_and_jacobian(configuration)
        return self._assemble_qp(error, jacobian, configuration._eye_nv)

    def compute_qp_residual(
        self, configuration: Configuration
    ) -> tuple[np.ndarray, np.ndarray, float]:
        error, jacobian = self._error_and_jacobian(configuration)
        return self._weighted_residual(error, jacobian)
