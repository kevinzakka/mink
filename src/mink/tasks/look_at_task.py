"""Look-at task implementation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..exceptions import InvalidTarget, TargetNotSet, TaskDefinitionError
from ..lie.utils import get_epsilon, skew
from .task import Objective, Task


class LookAtTask(Task):
    r"""Point a frame's axis at a target position in the world.

    Unlike a :class:`FrameTask` with an orientation target, this task only
    constrains *where the frame looks*, not how it is rolled about that direction.
    Pointing an axis at a point has two degrees of freedom (azimuth and
    elevation), so the task error and Jacobian both have rank two: rotation about
    the gaze axis is left free for other tasks to use.

    A frame :math:`B` is given a fixed unit *gaze axis* :math:`a` expressed in its
    own coordinates (e.g. the optical axis :math:`(0, 0, -1)` of a MuJoCo camera).
    Denoting the frame rotation and position in the world by :math:`R` and
    :math:`p`, and the target point by :math:`p^*`, the target direction in the
    frame is

    .. math::

        r = R^T (p^* - p), \quad n = \|r\|, \quad \hat{d} = r / n,

    and the task error is the misalignment of the gaze axis with that direction:

    .. math::

        e(q) = a - \hat{d}.

    Differentiating :math:`\hat{d}` with respect to the configuration, using the
    body Jacobian :math:`{}_B J_{WB} = [J_v; J_\omega]` returned by
    :meth:`Configuration.get_frame_jacobian` (both blocks expressed in the local
    frame), gives the task Jacobian

    .. math::

        J(q) = \frac{1}{n} (I - \hat{d}\,\hat{d}^T)
               \left( J_v - [r]_\times J_\omega \right),

    where :math:`[r]_\times` is the skew-symmetric matrix of :math:`r`. The
    projector :math:`(I - \hat{d}\,\hat{d}^T)` removes the component along the line
    of sight, which is what makes the task rank two. The translation block
    :math:`J_v` is included so the task accounts for the frame *moving* relative to
    the target, not just rotating: inverse kinematics may translate the frame to
    keep the target in view.

    The task is degenerate when the target coincides with the frame origin
    (:math:`n \to 0`) or lies directly behind the gaze axis (:math:`\hat{d} = -a`);
    in both cases the Levenberg-Marquardt ``lm_damping`` keeps the step bounded.

    Attributes:
        frame_name: Name or id of the frame to control, typically a body, geom or site.
        frame_type: The frame type: ``body``, ``geom`` or ``site``.
        axis: Unit gaze axis in the frame's local coordinates.
        target_pos: Target position to look at, in the world frame.

    Example:

    .. code-block:: python

        # Point a camera site's optical axis (-z) at a moving target.
        look_at_task = LookAtTask(
            frame_name="wrist_cam",
            frame_type="site",
            axis=(0.0, 0.0, -1.0),
            cost=1.0,
        )
        look_at_task.set_target(np.array([0.5, 0.0, 0.5]))
    """

    k: int = 3
    target_pos: np.ndarray | None

    def __init__(
        self,
        frame_name: str | int,
        frame_type: str,
        axis: npt.ArrayLike = (0.0, 0.0, 1.0),
        cost: npt.ArrayLike = 1.0,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        """Constructor.

        Args:
            frame_name: Name or id of the frame to control.
            frame_type: The frame type: ``body``, ``geom`` or ``site``.
            axis: Gaze axis in the frame's local coordinates. Need not be
                normalized; it is normalized internally. Defaults to the local
                ``+z`` axis.
            cost: Cost of the look-at task. A scalar or a vector of shape (1,).
            gain: Task gain in [0, 1] for additional low-pass filtering.
            lm_damping: Levenberg-Marquardt damping for ill-conditioned targets.
        """
        super().__init__(cost=np.zeros((self.k,)), gain=gain, lm_damping=lm_damping)
        self.frame_name = frame_name
        self.frame_type = frame_type
        self.target_pos = None

        self.set_axis(axis)
        self.set_cost(cost)

    def set_axis(self, axis: npt.ArrayLike) -> None:
        """Set the local gaze axis, normalizing to unit length."""
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
        """Set the cost of the look-at task.

        Args:
            cost: A scalar, or a vector of shape (1,). The look-at error is a
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

    def set_target(self, target_pos: npt.ArrayLike) -> None:
        """Set the target position to look at, in the world frame.

        Args:
            target_pos: A vector of shape (3,).
        """
        target_pos = np.atleast_1d(np.asarray(target_pos, dtype=float))
        if target_pos.shape != (self.k,):
            raise InvalidTarget(
                f"Expected target position to have shape ({self.k},) but got "
                f"{target_pos.shape}"
            )
        self.target_pos = target_pos.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set the target to a point one unit along the current gaze axis.

        This makes the task a no-op at the given configuration, useful for
        initialization.

        Args:
            configuration: Robot configuration :math:`q`.
        """
        transform = configuration.get_transform_frame_to_world(
            self.frame_name, self.frame_type
        )
        self.set_target(transform.apply(self.axis))

    def _error_and_jacobian(
        self, configuration: Configuration
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.target_pos is None:
            raise TargetNotSet(self.__class__.__name__)

        transform = configuration.get_transform_frame_to_world(
            self.frame_name, self.frame_type
        )
        rotation = transform.rotation().as_matrix()
        position = transform.translation()
        jac = configuration.get_frame_jacobian(self.frame_name, self.frame_type)
        jac_v, jac_w = jac[:3], jac[3:]

        # Target direction expressed in the local frame.
        r = rotation.T @ (self.target_pos - position)
        n = np.linalg.norm(r)
        n = max(n, get_epsilon(r.dtype))
        d_hat = r / n

        error = self.axis - d_hat
        projector = np.eye(3) - np.outer(d_hat, d_hat)
        jacobian = (projector @ (jac_v - skew(r) @ jac_w)) / n
        return error, jacobian

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the look-at task error.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Look-at task error :math:`e(q) = a - \hat{d}`.
        """
        error, _ = self._error_and_jacobian(configuration)
        return error

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the look-at task Jacobian.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Look-at task Jacobian :math:`J(q)`.
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
