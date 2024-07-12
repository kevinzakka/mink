import mujoco
import numpy as np

from ..configuration import Configuration
from .limit import Constraint, Limit


class VelocityLimit(Limit):
    def __init__(self, model: mujoco.MjModel, limit: np.ndarray):
        """Initialize velocity limits.

        Args:
            limit: Array of maximum allowed magnitudes of joint velocities for each
            joint, in m/s for slide joints and rad/s for hinge joints. The array should
            be ordered according to the joint indices in the robot model.
        """
        self.limit = limit
        self.projection_matrix = np.eye(model.nv)

    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        del configuration  # Unused.
        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        h = np.hstack([dt * self.limit, dt * self.limit])
        return Constraint(G=G, h=h)
