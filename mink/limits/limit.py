import abc
from typing import NamedTuple

import numpy as np

from ..configuration import Configuration


class Constraint(NamedTuple):
    """Linear inequalities in the form Gx <= h."""

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
            configuration: Configuration instance.
            dt: Integration time step in seconds.

        Returns:
            An instance of Constraint representing an inequality constraint of the form
            Gx <= h.
        """
        raise NotImplementedError
