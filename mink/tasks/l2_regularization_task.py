"""L2 regularization task implementation."""

import numpy as np

from ..configuration import Configuration
from .task import Objective, RegularizationTask


class L2RegularizationTask(RegularizationTask):
    r"""L2 regularization task.

    This low-priority task adds a Tikhonov/Levenberg-Marquardt term to the
    quadratic program, making the Hessian strictly positive-definite and
    selecting the **minimum-norm joint velocity** in any redundant or
    near-singular situation. Formally it contributes

    .. math::
        \tfrac12\, \lambda\, \dot{\mathbf{q}}^\top \dot{\mathbf{q}},

    where :math:`\dot{\mathbf{q}}` is the vector of joint velocities and
    :math:`\lambda` is the scalar strength of the regularization. Setting a larger
    value of :math:`\lambda` *reduces* motion in all DoFs; with no other active tasks
    the robot simply remains at rest.

    .. note::

        Unlike the :class:`~.DampingTask`, this task adds uniform damping to all DoFs,
        including floating-base coordinates. This is equivalent to setting the damping
        parameter in :func:`~.solve_ik`.
    """

    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        H = np.eye(configuration.nv)
        c = np.zeros(configuration.nv)
        return Objective(self.cost * H, c)
