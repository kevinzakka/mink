import numpy as np

from mink.limits import Limit, Constraint


class VelocityLimit(Limit):
    def __init__(self, limit: np.ndarray):
        """Initialize velocity limits.

        Args:
            limit: Array of maximum allowed magnitudes of joint velocities for each
            joint, in m/s for slide joints and rad/s for hinge joints. The array should
            be ordered according to the joint indices in the robot model.
        """
        self.limit = limit

    def compute_qp_inequalities(
        self,
        q: np.ndarray,
        dt: float,
    ) -> Constraint:
        del q  # Unused.

        # G = np.vstack([self.projection_matrix, -self.projection_matrix])
        # h = np.hstack([dt * v_max, dt * v_max])

        return Constraint(lower=-dt * self.limit, upper=dt * self.limit)
