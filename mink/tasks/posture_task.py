"""Posture task implementation."""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..utils import get_freejoint_dims
from .exceptions import InvalidTarget, TargetNotSet, TaskDefinitionError
from .task import Task


class PostureTask(Task):
    """Regulate the joint angles of the robot towards a desired posture.

    Using this task with a low cost value is useful as a regularizer, biasing the
    solution towards the desired posture when the problem is under-constrained.

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

    target_q: Optional[np.ndarray]

    def __init__(
        self,
        model: mujoco.MjModel,
        cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        self.target_q = None

        v_ids = np.arange(model.nv)
        _, v_ids_or_none = get_freejoint_dims(model)
        if v_ids_or_none is not None:
            v_ids = np.setdiff1d(v_ids, v_ids_or_none)

        self._v_ids = v_ids
        self.nv = model.nv
        self.k = len(v_ids)
        self.nq = model.nq

        super().__init__(
            cost=np.zeros((self.k,)),
            gain=gain,
            lm_damping=lm_damping,
        )
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
        c = cost[0] if cost.shape[0] == 1 else cost
        self.cost[:] = c

    def set_target(self, target_q: npt.ArrayLike) -> None:
        """Set the target posture.

        Args:
            target_q: Desired joint configuration.
        """
        target_q = np.atleast_1d(target_q)
        if target_q.ndim != 1 or target_q.shape[0] != (self.nq):
            raise InvalidTarget(
                f"Expected target posture to have shape ({self.nq},) but got "
                f"{target_q.shape}"
            )
        self.target_q = target_q.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set the target posture from the current configuration.

        Args:
            configuration: Robot configuration :math:`q`.
        """
        self.set_target(configuration.q)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the posture task error.

        The error is defined as:

        .. math::

            e(q) = q^* \ominus q

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

        return qvel[self._v_ids]

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the posture task Jacobian.

        The task Jacobian is defined as:

        .. math::

            J(q) = I_{n_v}

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Posture task jacobian :math:`J(q)`.
        """
        return configuration.eye[self._v_ids]
