"""Kinetic energy regularization task implementation."""

import mujoco
import numpy as np

from ..configuration import Configuration
from .task import Objective, BaseTask
from ..exceptions import TaskDefinitionError


class KineticEnergyRegularizationTask(BaseTask):
    r"""Kinetic-energy regularization task.

    This low-priority task adds a configuration-dependent quadratic term to the
    QP objective that penalizes the system's kinetic energy. Formally, it contributes:

    .. math::
        \tfrac12\, \lambda\, \dot{\mathbf{q}}^\top M(\mathbf{q})\, \dot{\mathbf{q}},

    where :math:`\dot{\mathbf{q}}` is the vector of joint velocities,
    :math:`M(\mathbf{q})` is the joint-space inertia matrix (dependent on the current
    configuration), and :math:`\lambda` is the scalar strength of the regularization.

    .. note::

        This task penalizes DoFs in proportion to their joint-space inertia, so
        higher-inertia (i.e., heavier) links will move less. This is in contrast to
        :class:`~.L2RegularizationTask`, which uniformly damps all DoFs.
    """

    def __init__(self, cost: float):
        if not np.isscalar(cost):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} cost must be a scalar"
            )
        if cost < 0:
            raise TaskDefinitionError(f"{self.__class__.__name__} cost should be >= 0")
        self.cost = cost

    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        mujoco.mj_crb(configuration.model, configuration.data)
        M = np.empty((configuration.nv, configuration.nv), dtype=np.float64)
        mujoco.mj_fullM(configuration.model, M, configuration.data.qM)
        return Objective(self.cost * M, np.zeros(configuration.nv))
