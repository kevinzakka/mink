from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from mink.limits import Limit, BoxConstraint


@dataclass(frozen=True)
class VelocityLimit(Limit):
    """Velocity limits."""

    limit: np.ndarray
    """Array of maximum allowed magnitudes of joint velocities for each joint, in
    m/s for slide joints and rad/s for hinge joints. The array should be ordered
    according to the joint indices in the robot model."""

    def compute_qp_inequalities(
        self,
        q: np.ndarray,
        dt: float,
    ) -> BoxConstraint:
        del q  # Unused.
        return BoxConstraint(lower=-dt * self.limit, upper=dt * self.limit)
