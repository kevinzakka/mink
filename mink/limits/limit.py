"""All kinematic limits derive from the :class:`Limit` base class."""

import abc
from typing import NamedTuple, Optional

import numpy as np

from ..configuration import Configuration


class BoxConstraint(NamedTuple):
    """Box constraint of the form lower <= x <= upper."""

    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None

    @property
    def inactive(self) -> bool:
        """Returns True if the constraint is inactive."""
        return self.lower is None and self.upper is None


class Limit(abc.ABC):
    """Abstract base class for kinematic limits.

    Subclasses must implement the :py:meth:`~Limit.compute_qp_inequalities` method
    which takes in the current robot configuration and integration time step and
    returns an instance of :class:`Constraint`.
    """

    @abc.abstractmethod
    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> BoxConstraint:
        raise NotImplementedError
