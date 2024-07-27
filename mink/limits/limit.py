"""All kinematic limits derive from the :class:`Limit` base class."""

import abc
from typing import NamedTuple, Optional

import numpy as np

from ..configuration import Configuration


class Constraint(NamedTuple):
    r"""Linear inequality constraint of the form :math:`G(q) \Delta q \leq h(q)`.

    Inactive if G and h are None.
    """

    G: Optional[np.ndarray] = None
    """Shape (nv, nv)."""
    h: Optional[np.ndarray] = None
    """Shape (nv,)."""

    @property
    def inactive(self) -> bool:
        """Returns True if the constraint is inactive."""
        return self.G is None and self.h is None


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
    ) -> Constraint:
        r"""Compute limit as linearized QP inequalities of the form:

        .. math::

            G(q) \Delta q \leq h(q)

        where :math:`q \in {\cal C}` is the robot's configuration and
        :math:`\Delta q \in T_q({\cal C})` is the displacement in the tangent
        space at :math:`q`.

        Args:
            configuration: Robot configuration :math:`q`.
            dt: Integration time step in [s].

        Returns:
            Pair :math:`(G, h)`.
        """
        raise NotImplementedError
