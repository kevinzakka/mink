"""Damping task implementation."""

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from .posture_task import PostureTask


class DampingTask(PostureTask):
    r"""L2-regularization on joint velocities (a.k.a. *velocity damping*).

    This low-priority task adds a Tikhonov/Levenberg-Marquardt term to the
    quadratic program, making the Hessian strictly positive-definite and
    selecting the **minimum-norm joint velocity** in any redundant or
    near-singular situation. Formally it contributes

    .. math::
        \tfrac12 \sum_{i=1}^{n_v} \lambda_i \dot q_i^{\,2},

    where ``cost`` provides the weights :math:`\lambda_i`. Setting a larger
    weight *reduces* motion in that DoF; with no other active tasks the robot
    simply remains at rest.

    This task does not favor a particular posture—only small instantaneous
    motion. If you need a posture bias, use an explicit :class:`~.PostureTask`.

    .. note::

        Floating-base coordinates are not affected by this task.

    Example:

    .. code-block:: python

        # Uniform damping across all degrees of freedom.
        damping_task = DampingTask(model, cost=1.0)

        # Custom damping.
        cost = np.zeros(model.nv)
        cost[:3] = 1.0  # High damping for the first 3 joints.
        cost[3:] = 0.1  # Lower damping for the remaining joints.
        damping_task = DampingTask(model, cost)
    """

    def __init__(self, model: mujoco.MjModel, cost: npt.ArrayLike):
        super().__init__(model, cost, gain=0.0, lm_damping=0.0)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the damping task error.

        The damping task does not chase a reference; its desired joint velocity
        is identically **zero**, so the error vector is always

        .. math:: e = \mathbf 0 \in \mathbb R^{n_v}.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Zero vector of length :math:`n_v`.
        """
        return np.zeros(configuration.nv)
