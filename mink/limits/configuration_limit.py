import numpy as np
import mujoco

from mink.limits import Limit, BoxConstraint

_SUPPORTED_JOINT_TYPES = {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}


class ConfigurationLimit(Limit):
    def __init__(self, model: mujoco.MjModel, limit_gain: float = 0.5):
        """Initialize configuration limits.

        Args:
            model: MuJoCo model.
            limit_gain: Gain factor between 0 and 1 that determines the percentage of
                maximum velocity allowed in each timestep.
        """
        if not 0.0 < limit_gain < 1.0:
            raise ValueError("Limit gain must be in the range (0, 1).")

        lower = np.full(model.nq, -np.inf)
        upper = np.full(model.nq, np.inf)
        for jnt in range(model.njnt):
            jnt_type = model.jnt_type[jnt]
            if jnt_type not in _SUPPORTED_JOINT_TYPES or not model.jnt_limited[jnt]:
                continue
            padr = model.jnt_qposadr[jnt]
            lower[padr : padr + 1] = model.jnt_range[jnt, 0]
            upper[padr : padr + 1] = model.jnt_range[jnt, 1]

        free_indices = []
        for jnt in range(model.njnt):
            if model.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_FREE:
                vadr = model.jnt_dofadr[jnt]
                free_indices.extend(np.arange(vadr, vadr + 7))
        free_indices = np.asarray(free_indices, dtype=int)

        self.free_indices = free_indices
        self.lower = lower
        self.upper = upper
        self.model = model
        self.limit_gain = limit_gain

    def compute_qp_inequalities(
        self,
        q: np.ndarray,
        dt: float,
    ) -> BoxConstraint:
        del dt  # Unused.

        delta_q_max = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(
            m=self.model,
            qvel=delta_q_max,
            dt=1.0,
            qpos1=q,
            qpos2=self.upper,
        )
        delta_q_max[self.free_indices] = np.inf

        delta_q_min = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(
            m=self.model,
            qvel=delta_q_min,
            dt=1.0,
            qpos1=q,
            qpos2=self.lower,
        )
        delta_q_min[self.free_indices] = -np.inf

        return BoxConstraint(
            lower=self.limit_gain * delta_q_min,
            upper=self.limit_gain * delta_q_max,
        )
