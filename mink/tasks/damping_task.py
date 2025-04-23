"""Damping task implementation."""

from __future__ import annotations

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from .posture_task import PostureTask


class DampingTask(PostureTask):
    r"""Minimize joint velocities.

    This damping task serves as a regularizer that minimizes the L2 norm of the joint
    velocities. This biases the solution toward the current configuration. A higher
    damping cost discourages motion and brings the robot to a stop if no other tasks
    are active.

    This task contributes the following quadratic penalty to the QP objective:

    .. math::
        \sum_i \lambda_i^2 \dot{q}_i^2,

    which acts as a weighted L2 regularization on joint velocities. The weight term
    :math:`\lambda_i` can be a scalar or a vector of shape ``(model.nv)``.

    Note: This task is implemented as a special case of :class:`PostureTask` where
    ``gain`` and ``lm_damping`` are set to zero.
    """

    def __init__(self, model: mujoco.MjModel, cost: npt.ArrayLike):
        super().__init__(model=model, cost=cost, gain=0.0, lm_damping=0.0)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        """Compute the damping task error."""
        del configuration  # Unused.
        return np.zeros((self.k,))
