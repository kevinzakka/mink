from typing import Optional, NamedTuple

import abc

import numpy as np


class Inequality(NamedTuple):
    """Linear inequality constraint in the form Gx <= h."""

    G: Optional[np.ndarray] = None
    h: Optional[np.ndarray] = None

    def inactive(self) -> bool:
        return self.G is None


class Limit(abc.ABC):
    """Abstract base class for kinematic limits."""

    @abc.abstractmethod
    def compute_qp_inequalities(self, q: np.ndarray, dt: float) -> Inequality:
        """Compute limit as linearized QP inequalities.

        Args:
            q: Configuration of the robot.
            dt: Integration time step in seconds.

        Returns:
            An instance of Inequality representing the inequality constraint as
            Gx <= h.
        """
