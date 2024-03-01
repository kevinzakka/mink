import abc
from typing import NamedTuple

import numpy as np

from mink.configuration import Configuration


class Objective(NamedTuple):
    """Quadratic objective function in the form 0.5 x^T H x + c^T x."""

    H: np.ndarray
    c: np.ndarray


class Task(abc.ABC):
    """Abstract base class for kinematic tasks.

    Subclasses must implement `compute_error` and `compute_jacobian` methods.
    """

    cost: np.ndarray | float | None
    gain: float
    lm_damping: float

    @abc.abstractmethod
    def compute_error(self, configuration: Configuration) -> np.ndarray:
        """Compute the task error function at the current configuration."""

    @abc.abstractmethod
    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """Compute the task Jacobian at the current configuration."""

    def _construct_weight(self, n: int) -> np.ndarray:
        if self.cost is None:
            return np.eye(n)
        diag = [self.cost] * n if isinstance(self.cost, float) else self.cost
        return np.diag(diag)

    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        jacobian = self.compute_jacobian(configuration)
        minus_gain_error = -self.gain * self.compute_error(configuration)
        weight = self._construct_weight(jacobian.shape[0])
        weighted_jacobian = weight @ jacobian
        weighted_error = weight @ minus_gain_error
        mu = self.lm_damping * weighted_error @ weighted_error
        eye_tg = np.eye(configuration.model.nv)
        H = weighted_jacobian.T @ weighted_jacobian + mu * eye_tg
        c = -weighted_error.T @ weighted_jacobian
        return Objective(H, c)
