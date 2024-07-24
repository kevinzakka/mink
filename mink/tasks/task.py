import abc
from typing import NamedTuple

import numpy as np

from ..configuration import Configuration
from .exceptions import InvalidDamping, InvalidGain


class Objective(NamedTuple):
    """Quadratic objective function in the form 0.5 x^T H x + c^T x."""

    H: np.ndarray  # (nv, nv)
    c: np.ndarray  # (nv,)

    def value(self, x: np.ndarray) -> float:
        """Returns the value of the objective at the input vector."""
        return x.T @ self.H @ x + self.c @ x


class Task(abc.ABC):
    """Abstract base class for kinematic tasks."""

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

        if not lm_damping >= 0.0:
            raise InvalidDamping("`lm_damping` must be >= 0")

        self.cost = cost
        self.gain = gain
        self.lm_damping = lm_damping

    @abc.abstractmethod
    def compute_error(self, configuration: Configuration) -> np.ndarray:
        """Compute the task error function at the current configuration.

        The error function e(q), of shape (k,), is the quantity that the task aims to
        drive to zero. It appears in the first-order task dynamics:

            J(q) Δq = -alpha e(q)

        The Jacobian matrix J(q), of shape (k, nv), is the derivative of the task error
        with respect to the configuration. This Jacobian is implemented in
        `compute_jacobian`. Finally, the configuration displacement Δq is the output
        of inverse kinematics.

        In the first-order task dynamics, the error e(q) is multiplied by the task gain
        alpha which is between [0, 1]. This gain can be 1.0 for dead-beat control
        (i.e., converge as fast as possible), but might be unstable as it neglects the
        first-order approximation. Lower values make the task slower and have a similar
        effect to low-pass filtering.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """Compute the task Jacobian at the current configuration.

        The task Jacobian J(q), of shape (k, nv) is the first-order derivative of the
        error e(q) that defines the task, with k the dimension of the task and
        nv the dimension of the robot's tangent space.
        """
        raise NotImplementedError

    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        """Compute the matrix-vector pair :math:`(H, c)` of the QP objective.

        This pair is such that the contribution of the task to the QP objective
        of the IK is:

        The weight matrix W, of shape (k, k), weighs and normalizes task coordinates
        to the same unit. The unit of the overall contribution is [cost]^2.
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
