from typing import Optional, NamedTuple

import abc

import numpy as np


class BoxConstraint(NamedTuple):
    """Box constraint of the form lower <= x <= upper."""

    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None

    def inactive(self) -> bool:
        return self.lower is None and self.upper is None


# class Constraint(NamedTuple):

#     G: np.ndarray
#     h: np.ndarray


class Limit(abc.ABC):
    """Abstract base class for kinematic limits."""

    @abc.abstractmethod
    def compute_qp_inequalities(
        self,
        q: np.ndarray,
        dt: float,
    ) -> BoxConstraint:
        """Compute limit as a box constraint.

        Args:
            q: Configuration of the robot.
            dt: Integration time step in seconds.

        Returns:
            An instance of BoxConstraint representing box constraints of the form
            lower <= x <= upper.
        """
