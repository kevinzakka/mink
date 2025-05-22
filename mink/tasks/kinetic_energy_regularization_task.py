"""Kinetic energy regularization task implementation."""

from __future__ import annotations

from typing import Optional, SupportsFloat

import numpy as np

from ..configuration import Configuration
from ..exceptions import IntegrationTimestepNotSet, TaskDefinitionError
from .task import BaseTask, Objective


class KineticEnergyRegularizationTask(BaseTask):
    r"""Kinetic-energy regularization task.

    This low-priority task adds a configuration-dependent quadratic term to the
    QP objective that penalizes the system's kinetic energy. Formally, it contributes:

    .. math::
        \frac{1}{2}\, \lambda\, \Delta \mathbf{q}^\top M(\mathbf{q})\,\Delta \mathbf{q},

    where :math:`\Delta \mathbf{q}\in\mathbb{R}^{n_v}` is the vector of joint
    displacements, :math:`M(\mathbf{q})` is the joint-space inertia matrix
    (dependent on the current configuration), and :math:`\lambda` is the scalar
    strength of the regularization.

    .. note::

        This task penalizes DoFs in proportion to their joint-space inertia, so
        higher-inertia (i.e., heavier) links will move less. This is in contrast to
        :class:`~.L2RegularizationTask`, which uniformly damps all DoFs.

    .. warning::

        The timestep :math:`\Delta t` must be set via :meth:`set_dt` before use.
        This task penalizes velocity via displacement, and omitting `dt` will raise
        a runtime error.

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
        self.inv_dt_sq: Optional[float] = None

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
