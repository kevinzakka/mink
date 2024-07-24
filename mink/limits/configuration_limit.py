"""Limit on joint positions.

Derivation
==========

Using a first order Taylor expansion on the configuration, we can write the limit as:

    q_min     <= q + v * dt <= q_max
    q_min     <= q + dq     <= q_max
    q_min - q <= dq         <= q_max - q

Rewriting as G dq <= h:

    +I * dq <= q_max - q
    -I * dq <= q - q_min

Stacking them together, we get:

    G = [+I, -I]
    h = [q_max - q, q - q_min]
"""

import mujoco
import numpy as np

from ..configuration import Configuration
from ..constants import qpos_width
from .exceptions import LimitDefinitionError
from .limit import Constraint, Limit


class ConfigurationLimit(Limit):
    """Limit for joint positions in a model.

    Floating base joints (joint type="free") are ignored.

    Attributes:
        indices: Tangent indices corresponding to configuration-limited joints.
        projection_matrix: Projection from tangent space to subspace with
            configuration-limited joints.
        lower: Lower configuration limit.
        upper: Upper configuration limit.
    """

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
            raise LimitDefinitionError(
                f"{self.__class__.__name__} gain must be in the range (0, 1]"
            )

        index_list: list[int] = []
        lower = np.full(model.nq, -np.inf)
        upper = np.full(model.nq, np.inf)
        for jnt in range(model.njnt):
            jnt_type = model.jnt_type[jnt]
            qpos_dim = qpos_width(jnt_type)
            jnt_range = model.jnt_range[jnt]
            padr = model.jnt_qposadr[jnt]
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE or not model.jnt_limited[jnt]:
                continue
            lower[padr : padr + qpos_dim] = jnt_range[0] + min_distance_from_limits
            upper[padr : padr + qpos_dim] = jnt_range[1] - min_distance_from_limits
            index_list.append(jnt)

        self.indices = np.array(index_list)
        self.indices.setflags(write=False)

        dim = len(self.indices)
        self.projection_matrix = np.eye(model.nv)[self.indices] if dim > 0 else None

        self.lower = lower
        self.upper = upper
        self.model = model
        self.gain = gain

    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        del dt  # Unused.

        # Upper.
        delta_q_max = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(
            m=self.model,
            qvel=delta_q_max,
            dt=1.0,
            qpos1=configuration.q,
            qpos2=self.upper,
        )

        # Lower.
        delta_q_min = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(
            m=self.model,
            qvel=delta_q_min,
            dt=1.0,
            # NOTE: mujoco.mj_differentiatePos does `qpos2 - qpos1` so notice the order
            # swap here compared to above.
            qpos1=self.lower,
            qpos2=configuration.q,
        )

        p_min = self.gain * delta_q_min[self.indices]
        p_max = self.gain * delta_q_max[self.indices]
        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        h = np.hstack([p_max, p_min])
        return Constraint(G=G, h=h)
