from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from mink.limits import BoxConstraint


@dataclass(frozen=True)
class AccelerationLimit:
    """Acceleration limits."""

    limits: np.ndarray
    """Array of maximum allowed magnitudes of joint accelerations for each joint, in
    m/s^2 for slide joints and rad/s^2 for hinge joints. The array should be ordered
    according to the joint indices in the robot model."""

    def compute_qp_inequalities(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        dt: float,
    ) -> BoxConstraint:
        del q  # Unused.
        lower = dq - dt * self.limits
        upper = dq + dt * self.limits
        np.testing.assert_array_less(lower, upper)
        return BoxConstraint(lower=lower, upper=upper)
