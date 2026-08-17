"""Kinetic energy regularization task implementation."""

from __future__ import annotations

from typing import SupportsFloat

import numpy as np

from ..configuration import Configuration
from ..exceptions import IntegrationTimestepNotSet, TaskDefinitionError
from .task import BaseTask, Objective


class KineticEnergyRegularizationTask(BaseTask):
    r"""Kinetic-energy regularization.

    This task, often used with a low priority in the task stack, penalizes the system's
    kinetic energy. Formally, it contributes the following term to the quadratic
    program:

    .. math::
        \frac{1}{2}\, \lambda\, \Delta \mathbf{q}^\top \frac{M(\mathbf{q})}{(\Delta t)^2}\,\Delta \mathbf{q},

    where :math:`\Delta \mathbf{q}\in\mathbb{R}^{n_v}` is the vector of joint
    displacements, :math:`M(\mathbf{q})` is the joint-space inertia matrix,
    :math:`\lambda` is the scalar strength of the regularization, and
    :math:`\Delta t` is the integration timestep.

    .. note::

        This task can be seen as an inertia-weighted version of the
        :class:`~.DampingTask`. Degrees of freedom with higher inertia will move less
        for the same cost. Conversely, low-inertia degrees of freedom are nearly free
        to move, so pair this task with a small uniform :class:`~.DampingTask` to
        keep them from absorbing all the motion.

    .. warning::

        The integration timestep :math:`\Delta t` must be set via :meth:`set_dt`
        before use. This ensures the cost is expressed in units of energy (Joules).

    Example:

    .. code-block:: python

        task = KineticEnergyRegularizationTask(cost=1e-4)
        task.set_dt(0.02)
    """

    def __init__(self, cost: SupportsFloat):
        cost = float(cost)
        if cost < 0:
            raise TaskDefinitionError(f"{self.__class__.__name__} cost should be >= 0")
        self.cost: float = cost

        # Kinetic energy is defined as T = ½ * q̇ᵀ * M * q̇. Since q̇ ≈ Δq / dt, we
        # substitute: T ≈ ½ * (Δq / dt)ᵀ * M * (Δq / dt) = ½ * Δqᵀ * (M / dt²) * Δq.
        # Therefore, we scale the inertia matrix by 1 / dt² to penalize energy in
        # terms of joint displacements Δq.
        self.inv_dt_sq: float | None = None

    def set_dt(self, dt: float) -> None:
        """Set the integration timestep.

        Args:
            dt: Integration timestep in [s].
        """
        self.inv_dt_sq = 1.0 / dt**2

    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        if self.inv_dt_sq is None:
            raise IntegrationTimestepNotSet(self.__class__.__name__)
        M = configuration.get_inertia_matrix()
        H = self.cost * self.inv_dt_sq * M
        c = np.zeros(configuration.nv)
        return Objective(H, c)
