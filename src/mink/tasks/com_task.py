"""Center-of-mass task implementation."""

from __future__ import annotations

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..exceptions import InvalidTarget, TargetNotSet, TaskDefinitionError
from .task import DEFAULT_ELASTIC_PENALTY, Task


class ComTask(Task):
    """Regulate the center-of-mass (CoM) of a robot.

    Attributes:
        target_com: Target position of the CoM.
        elastic: If True, the task is an exact L1 penalty that holds like a
            constraint and yields when infeasible. See :class:`~mink.Task`.
        penalty: Per-component L1 penalty of an elastic task, a scalar or a
            vector of shape (3,).

    Example:

    .. code-block:: python

        com_task = ComTask(model, cost=1.0)

        # Update the target CoM directly.
        com_desired = np.zeros(3)
        com_task.set_target(com_desired)

        # Or from a keyframe defined in the model.
        configuration.update_from_keyframe("home")
        com_task.set_target_from_configuration(configuration)
    """

    k: int = 3
    target_com: np.ndarray | None

    def __init__(
        self,
        cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
        elastic: bool = False,
        penalty: npt.ArrayLike = DEFAULT_ELASTIC_PENALTY,
    ):
        super().__init__(
            cost=np.zeros((self.k,)),
            gain=gain,
            lm_damping=lm_damping,
            elastic=elastic,
            penalty=penalty,
        )
        self.target_com = None

        self.set_cost(cost)

    def set_cost(self, cost: npt.ArrayLike) -> None:
        """Set the cost of the CoM task.

        Args:
            cost: A vector of shape (1,) (aka identical cost for all coordinates),
                or (3,) (aka different costs for each coordinate).
        """
        cost = np.atleast_1d(cost)
        if cost.ndim != 1 or cost.shape[0] not in (1, self.k):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} cost must be a vector of shape (1,) "
                f"(aka identical cost for all coordinates) or ({self.k},). "
                f"Got {cost.shape}"
            )
        if not np.all(cost >= 0.0):
            raise TaskDefinitionError(f"{self.__class__.__name__} cost must be >= 0")
        self.cost[:] = cost

    def set_target(self, target_com: npt.ArrayLike) -> None:
        """Set the target CoM position in the world frame.

        Args:
            target_com: A vector of shape (3,) representing the desired
                center-of-mass position in the world frame.
        """
        target_com = np.atleast_1d(target_com)
        if target_com.ndim != 1 or target_com.shape[0] != (self.k):
            raise InvalidTarget(
                f"Expected target CoM to have shape ({self.k},) but got "
                f"{target_com.shape}"
            )
        self.target_com = target_com.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set the target CoM from a given robot configuration.

        Args:
            configuration: Robot configuration :math:`q`.
        """
        self.set_target(configuration.data.subtree_com[1])

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the CoM task error.

        The center of mass :math:`c(q)` for a collection of bodies :math:`\mathcal{B}`
        is the mass-weighted average of their individual centers of mass. After running
        forward kinematics, in particular after calling ``mj_comPos``, MuJoCo stores
        the CoM of each subtree in ``data.subtree_com``. This task uses the CoM of the
        subtree starting from body 1, which is the entire robot excluding the world
        body (body 0).

        The task error :math:`e(q)` is the difference between the current CoM
        :math:`c(q)` and the target CoM :math:`c^*`:

        .. math::

            e(q) = c(q) - c^*

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Center-of-mass task error vector :math:`e(q)`.
        """
        if self.target_com is None:
            raise TargetNotSet(self.__class__.__name__)
        # Note: body 0 is the world body, so we start from body 1 (the robot).
        # TODO(kevin): Don't hardcode subtree index.
        return configuration.data.subtree_com[1] - self.target_com

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the Jacobian of the CoM task error :math:`e(q)`.

        The Jacobian is the derivative of this error with respect to the
        generalized coordinates :math:`q`. Since the target :math:`c^*` is
        constant, the Jacobian of the error simplifies to the Jacobian of the
        CoM position :math:`c(q)`:

        .. math::

            J(q) = \frac{\partial e(q)}{\partial q} = \frac{\partial c(q)}{\partial q}

        MuJoCo's ``mj_jacSubtreeCom`` function computes this Jacobian using the
        formula:

        .. math::

            \frac{\partial c(q)}{\partial q} =
            \frac{1}{M} \sum_{i \in \mathcal{B}} m_i \frac{\partial p_i(q)}{\partial q}
            = \frac{1}{M} \sum_{i \in \mathcal{B}} m_i J_i(q)

        where :math:`M = \sum_{i \in \mathcal{B}} m_i` is the total mass of the subtree,
        :math:`m_i` is the mass of body :math:`i`, :math:`p_i(q)` is the position
        of the origin of body frame :math:`i` in the world frame, :math:`J_i(q) =
        \frac{\partial p_i(q)}{\partial q}` is the Jacobian mapping joint velocities to
        the linear velocity of the origin of body frame :math:`i`, and the sum is over
        all bodies :math:`\mathcal{B}` in the specified subtree (body 1 and its
        descendants).

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Jacobian of the center-of-mass task error :math:`J(q)`.
        """
        # NOTE: We don't need a target CoM to compute this Jacobian.
        jac = np.empty((self.k, configuration.nv))
        mujoco.mj_jacSubtreeCom(configuration.model, configuration.data, jac, 1)
        return jac
