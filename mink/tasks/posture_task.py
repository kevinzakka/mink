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

    A posture is a vector of actuated joint angles. Floating-base coordinates are not
    affected by this task.

    Attributes:
        target_q: Target configuration.
    """

    target_q: Optional[np.ndarray]

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
            qpos1=configuration.q,
            qpos2=self.target_q,
        )

        if self._v_ids is not None:
            qvel[self._v_ids] = 0.0

        return qvel

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
        if self.target_q is None:
            raise TargetNotSet(self.__class__.__name__)

        jac = -np.eye(configuration.nv)
        if self._v_ids is not None:
            jac[:, self._v_ids] = 0.0

        return jac
