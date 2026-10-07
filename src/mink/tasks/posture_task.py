"""Posture task implementation."""

from __future__ import annotations

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..exceptions import InvalidTarget, TargetNotSet, TaskDefinitionError
from ..lie import SO3
from ..utils import get_freejoint_dims
from .task import Task


class PostureTask(Task):
    """Regulate joint angles towards a target posture.

    Often used with a low priority in the task stack, this task acts like a regularizer,
    biasing the solution toward a specific joint configuration.

    Attributes:
        target_q: Target configuration :math:`q^*`, of shape :math:`(n_q,)`. Units are
            radians for revolute joints and meters for prismatic joints. Note that
            floating-base coordinates are not affected by this task but should be
            included in the target configuration.

    Example:

    .. code-block:: python

        posture_task = PostureTask(model, cost=1e-3)

        # Update the target posture directly.
        q_desired = ...
        posture_task.set_target(q_desired)

        # Or from a keyframe defined in the model.
        configuration.update_from_keyframe("home")
        posture_task.set_target_from_configuration(configuration)
    """

    target_q: np.ndarray | None

    def __init__(
        self,
        model: mujoco.MjModel,
        cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        super().__init__(
            cost=np.zeros((model.nv,)),
            gain=gain,
            lm_damping=lm_damping,
        )
        self.target_q = None

        self._v_ids: np.ndarray | None
        _, v_ids_or_none = get_freejoint_dims(model)
        if v_ids_or_none:
            self._v_ids = np.asarray(v_ids_or_none)
        else:
            self._v_ids = None

        self._ball_dofadr = model.jnt_dofadr[
            model.jnt_type == mujoco.mjtJoint.mjJNT_BALL
        ].copy()

        self.k = model.nv
        self.nq = model.nq
        self.set_cost(cost)

    def set_cost(self, cost: npt.ArrayLike) -> None:
        cost = np.atleast_1d(cost)
        if cost.ndim != 1 or cost.shape[0] not in (1, self.k):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} cost must be a vector of shape (1,) "
                f"(aka identical cost for all dofs) or ({self.k},). Got {cost.shape}"
            )
        if not np.all(cost >= 0.0):
            raise TaskDefinitionError(f"{self.__class__.__name__} cost should be >= 0")
        self.cost[: self.k] = cost

    def set_target(self, target_q: npt.ArrayLike) -> None:
        """Set the target posture.

        Args:
            target_q: A vector of shape (nq,) representing the desired joint
                configuration.
        """
        target_q = np.atleast_1d(target_q)
        if target_q.ndim != 1 or target_q.shape[0] != (self.nq):
            raise InvalidTarget(
                f"Expected target posture to have shape ({self.nq},) but got "
                f"{target_q.shape}"
            )
        self.target_q = target_q.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set the target posture by extracting it from the current configuration.

        Args:
            configuration: Robot configuration :math:`q`.
        """
        self.set_target(configuration.q)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the posture task error.

        The error is defined as:

        .. math::

            e(q) = q \ominus q^*

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Posture task error vector :math:`e(q)`.
        """
        if self.target_q is None:
            raise TargetNotSet(self.__class__.__name__)

        # NOTE: mj_differentiatePos calculates qpos2 ⊖ qpos1.
        qvel = np.empty(configuration.nv)
        mujoco.mj_differentiatePos(
            m=configuration.model,
            qvel=qvel,
            dt=1.0,
            qpos1=self.target_q,
            qpos2=configuration.q,
        )

        if self._v_ids is not None:
            qvel[self._v_ids] = 0.0

        return qvel

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the posture task Jacobian.

        Hinge and slide coordinates have an identity Jacobian. For a ball
        joint with error :math:`\phi = \log(R^{*T} R)`, MuJoCo integrates
        a local angular displacement on the right, giving the block
        :math:`J_r^{-1}(\phi)`. Floating-base columns remain zero.

        Before a target is set, the identity projection is returned as before.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Posture task jacobian :math:`J(q)`.
        """
        jac = np.eye(configuration.nv)
        if self.target_q is not None and len(self._ball_dofadr) > 0:
            error = self.compute_error(configuration)
            for dofadr in self._ball_dofadr:
                block = slice(dofadr, dofadr + 3)
                jac[block, block] = SO3.rjacinv(error[block])
        if self._v_ids is not None:
            jac[:, self._v_ids] = 0.0
        return jac
