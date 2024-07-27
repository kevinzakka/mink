"""All kinematic tasks derive from the :class:`Task` base class."""

import abc
from typing import NamedTuple

import numpy as np

from ..configuration import Configuration
from .exceptions import InvalidDamping, InvalidGain


class Objective(NamedTuple):
    r"""Quadratic objective of the form :math:`\frac{1}{2} x^T H x + c^T x`."""

    H: np.ndarray
    """Hessian matrix, of shape (n_v, n_v)"""
    c: np.ndarray
    """Linear vector, of shape (n_v,)."""

    def value(self, x: np.ndarray) -> float:
        """Returns the value of the objective at the input vector."""
        return x.T @ self.H @ x + self.c @ x


class Task(abc.ABC):
    r"""Abstract base class for kinematic tasks.

    Subclasses must implement the configuration-dependent task error
    :py:meth:`~Task.compute_error` and Jacobian :py:meth:`~Task.compute_jacobian`
    functions.

    The error function :math:`e(q) \in \mathbb{R}^{k}` is the quantity that
    the task aims to drive to zero (:math:`k` is the dimension of the
    task). It appears in the first-order task dynamics:

    .. math::

        J(q) \Delta q = -\alpha e(q)

    The Jacobian matrix :math:`J(q) \in \mathbb{R}^{k \times n_v}`, with
    :math:`n_v` the dimension of the robot's tangent space, is the
    derivative of the task error :math:`e(q)` with respect to the
    configuration :math:`q \in \mathbb{R}^{n_q}`. The configuration displacement
    :math:`\Delta q` is the output of inverse kinematics; we divide it by dt to get a
    commanded velocity.

    In the first-order task dynamics, the error :math:`e(q)` is multiplied
    by the task gain :math:`\alpha \in [0, 1]`. This gain can be 1.0 for
    dead-beat control (*i.e.* converge as fast as possible), but might be
    unstable as it neglects our first-order approximation. Lower values
    cause slow down the task, similar to low-pass filtering.
    """

    def __init__(
        self,
        cost: np.ndarray,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        """Constructor.

        Args:
            cost: Cost vector with the same dimension as the error of the task.
            gain: Task gain alpha in [0, 1] for additional low-pass filtering. Defaults
                to 1.0 (no filtering) for dead-beat control.
            lm_damping: Unitless scale of the Levenberg-Marquardt (only when the error
            is large) regularization term, which helps when targets are infeasible.
            Increase this value if the task is too jerky under unfeasible targets, but
            beware that a larger damping slows down the task.
        """
        if not 0.0 <= gain <= 1.0:
            raise InvalidGain("`gain` must be in the range [0, 1]")

        if lm_damping < 0.0:
            raise InvalidDamping("`lm_damping` must be >= 0")

        self.cost = cost
        self.gain = gain
        self.lm_damping = lm_damping

    @abc.abstractmethod
    def compute_error(self, configuration: Configuration) -> np.ndarray:
        """Compute the task error at the current configuration.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task error vector :math:`e(q)`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """Compute the task Jacobian at the current configuration.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task jacobian :math:`J(q)`.
        """
        raise NotImplementedError

    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        r"""Compute the matrix-vector pair :math:`(H, c)` of the QP objective.

        This pair is such that the contribution of the task to the QP objective is:

        .. math::

            \| J \Delta q + \alpha e \|_{W}^2 = \frac{1}{2} \Delta q^T H
            \Delta q + c^T q

        The weight matrix :math:`W \in \mathbb{R}^{k \times k}` weights and
        normalizes task coordinates to the same unit. The unit of the overall
        contribution is [cost]^2.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Pair :math:`(H(q), c(q))`.
        """
        jacobian = self.compute_jacobian(configuration)  # (k, nv)
        minus_gain_error = -self.gain * self.compute_error(configuration)  # (k,)

        weight = np.diag(self.cost)
        weighted_jacobian = weight @ jacobian
        weighted_error = weight @ minus_gain_error

        mu = self.lm_damping * weighted_error @ weighted_error
        eye_tg = np.eye(configuration.model.nv)

        H = weighted_jacobian.T @ weighted_jacobian + mu * eye_tg  # (nv, nv)
        c = -weighted_error.T @ weighted_jacobian  # (nv,)

        return Objective(H, c)
