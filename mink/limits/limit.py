from typing import Iterable, Optional

import abc

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Inequality:
    """Linear inequality constraint in the form Gx <= h."""

    G: Optional[np.ndarray] = None
    h: Optional[np.ndarray] = None

    def __iter__(self) -> Iterable[np.ndarray]:
        return iter((self.G, self.h))

    def inactive(self) -> bool:
        return self.G is None and self.h is None


class Limit(abc.ABC):
    """Abstract base class for kinematic limits."""

    @abc.abstractclassmethod
    def compute_qp_inequalities(self, q: np.ndarray, dt: float) -> Inequality:
        """Compute limit as linearized QP inequalities.

        Args:
            q: Configuration of the robot.
            dt: Integration time step in seconds.

        Returns:
            An instance of Inequality representing the inequality constraint as
            Gx <= h.
        """
