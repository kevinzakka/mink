import numpy as np
import mujoco

from mink.limits import Limit, Constraint
from mink.configuration import Configuration

_SUPPORTED_JOINT_TYPES = {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}


class ConfigurationLimit(Limit):
    def __init__(
        self,
        model: mujoco.MjModel,
        gain: float = 0.95,
        min_distance_from_limits: float = 0.0,
    ):
        """Initialize configuration limits.

        Args:
            model: MuJoCo model.
            gain: Gain factor in (0, 1] that determines how fast each joint is
                allowed to move towards the joint limits at each timestep. Values lower
                ttan 1 are safer but may make the joints move slowly.
            min_distance_from_limits: Offset in meters (slide joints) or radians
                (hinge joints) to be added to the limits. Positive values decrease the
                range of motion, negative values increase it (i.e. negative values
                allow penetration).
        """
        if not 0.0 < gain <= 1.0:
            raise ValueError("Limit gain must be in the range (0, 1].")

        indices = []
        lower = np.full(model.nq, -np.inf)
        upper = np.full(model.nq, np.inf)
        for jnt in range(model.njnt):
            jnt_type = model.jnt_type[jnt]
            if jnt_type not in _SUPPORTED_JOINT_TYPES or not model.jnt_limited[jnt]:
                continue
            padr = model.jnt_qposadr[jnt]
            lower[padr : padr + 1] = model.jnt_range[jnt, 0] + min_distance_from_limits
            upper[padr : padr + 1] = model.jnt_range[jnt, 1] - min_distance_from_limits
            indices.append(jnt)

        free_indices = []
        for jnt in range(model.njnt):
            if model.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_FREE:
                vadr = model.jnt_dofadr[jnt]
                free_indices.extend(np.arange(vadr, vadr + 7))
        free_indices = np.asarray(free_indices, dtype=int)

        self.projection_matrix = np.eye(model.nv)
        self.free_indices = free_indices
        self.lower = lower
        self.upper = upper
        self.model = model
        self.gain = gain
        self.indices = indices

    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        del dt  # Unused.

        delta_q_max = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(
            m=self.model,
            qvel=delta_q_max,
            dt=1.0,
            qpos1=configuration.q,
            qpos2=self.upper,
        )
        delta_q_max[self.free_indices] = np.inf

        delta_q_min = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(
            m=self.model,
            qvel=delta_q_min,
            dt=1.0,
            qpos1=configuration.q,
            qpos2=self.lower,
        )
        delta_q_min[self.free_indices] = -np.inf

        p_max = self.gain * delta_q_max
        p_min = self.gain * delta_q_min
        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        h = np.hstack([p_max, -p_min])
        return Constraint(G=G, h=h)
