import abc
from typing import NamedTuple

import numpy as np

from ..configuration import Configuration


class Constraint(NamedTuple):
    """Linear inequalities in the form G(q) dq <= h(q).

    The limit is considered inactive when G or h are None.
    """

    G: np.ndarray | None = None  # (nv, nv)
    h: np.ndarray | None = None  # (nv,)

    @property
    def inactive(self) -> bool:
        return self.G is None or self.h is None


class Limit(abc.ABC):
    """Abstract base class for kinematic limits."""

    @abc.abstractmethod
    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        """Compute limit as linearized QP inequalities.

        Args:
            configuration: Current configuration.
            dt: Integration time step in [s].

        Returns:
            Pair (G, h) representing the inequality constraint.
        """
        raise NotImplementedError
