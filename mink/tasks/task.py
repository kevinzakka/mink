import abc
from typing import Iterable

import numpy as np
from dataclasses import dataclass

from mink.configuration import Configuration


@dataclass(frozen=True)
class Objective:
    """Quadratic objective function in the form 0.5 x^T H x + c^T x."""

    H: np.ndarray
    c: np.ndarray

    def __iter__(self) -> Iterable[np.ndarray]:
        return iter((self.H, self.c))


class Task(abc.ABC):
    """Abstract base class for kinematic tasks.

    Subclasses must implement `compute_error` and `compute_jacobian` methods.
    """

    cost: np.ndarray
    gain: float
    lm_damping: float

    @abc.abstractmethod
    def compute_error(self, configuration: Configuration) -> np.ndarray:
        """Compute the task error function."""

    @abc.abstractmethod
    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """Compute the task Jacobian at the current configuration."""

    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        jacobian = self.compute_jacobian(configuration)
        gain_error = self.gain * self.compute_error(configuration)
        weight = (
            np.eye(jacobian.shape[0])
            if self.cost is None
            else np.diag(
                [self.cost] * jacobian.shape[0]
                if isinstance(self.cost, float)
                else self.cost
            )
        )
        weighted_jacobian = weight @ jacobian
        weighted_error = weight @ gain_error
        mu = self.lm_damping * weighted_error @ weighted_error
        eye_tg = np.eye(configuration.model.nv)
        H = weighted_jacobian.T @ weighted_jacobian + mu * eye_tg
        c = -weighted_error.T @ weighted_jacobian
        return Objective(H, c)
