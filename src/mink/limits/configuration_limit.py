"""Joint position limit."""

import mujoco
import numpy as np

from ..configuration import Configuration
from ..constants import dof_width, qpos_width
from ..exceptions import LimitDefinitionError
from .limit import Constraint, Limit


class ConfigurationLimit(Limit):
    """Inequality constraint on joint positions in a robot model.

    Floating base joints are ignored. Slide and hinge joints are box-constrained to
    their range. A limited ball joint is constrained on its total rotation angle,
    bounded by the second element of its range (the first element is ignored,
    following MuJoCo semantics).

    .. note::

        The ball joint constraint is a first-order linearization that only bounds
        motion along the current rotation axis; motion orthogonal to the axis grows
        the angle at second order. Pair it with a velocity limit on the ball
        joint's degrees of freedom so that large steps cannot overshoot the limit.
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
                (hinge and ball joints) to be added to the limits. Positive values
                decrease the range of motion, negative values increase it (i.e.
                negative values allow penetration).
        """
        if not 0.0 < gain <= 1.0:
            raise LimitDefinitionError(
                f"{self.__class__.__name__} gain must be in the range (0, 1]"
            )

        index_list: list[int] = []  # DoF indices that are limited.
        ball_jnt_list: list[int] = []  # Limited ball joints, handled separately.
        lower = np.full(model.nq, -mujoco.mjMAXVAL)
        upper = np.full(model.nq, mujoco.mjMAXVAL)
        for jnt in range(model.njnt):
            jnt_type = model.jnt_type[jnt]
            qpos_dim = qpos_width(jnt_type)
            jnt_range = model.jnt_range[jnt]
            padr = model.jnt_qposadr[jnt]
            # Skip free joints and joints without limits.
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE or not model.jnt_limited[jnt]:
                continue
            # Ball joints bound the rotation angle and are handled separately.
            if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                ball_jnt_list.append(jnt)
                continue
            lower[padr : padr + qpos_dim] = jnt_range[0] + min_distance_from_limits
            upper[padr : padr + qpos_dim] = jnt_range[1] - min_distance_from_limits
            jnt_dim = dof_width(jnt_type)
            jnt_id = model.jnt_dofadr[jnt]
            index_list.extend(range(jnt_id, jnt_id + jnt_dim))

        self._ball_qposadr = model.jnt_qposadr[ball_jnt_list]
        self._ball_dofadr = model.jnt_dofadr[ball_jnt_list]
        self._ball_max_angle = (
            model.jnt_range[ball_jnt_list, 1] - min_distance_from_limits
        )
        if np.any(self._ball_max_angle <= 0.0):
            raise LimitDefinitionError(
                f"{self.__class__.__name__} ball joint max rotation angle must be "
                "positive after subtracting min_distance_from_limits"
            )

        # Scratch buffers for the ball joint rows.
        n_ball = len(ball_jnt_list)
        self._ball_G = np.zeros((n_ball, model.nv))
        self._ball_h = np.empty(n_ball)
        self._phi = np.empty(3)

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
        r"""Compute the configuration-dependent joint position limits.

        The limits are defined as:

        .. math::

            {q \ominus q_{min}} \leq \Delta q \leq {q_{max} \ominus q}

        where :math:`q \in {\cal C}` is the robot's configuration and
        :math:`\Delta q \in T_q({\cal C})` is the displacement in the tangent
        space at :math:`q`. See the :ref:`derivations` section for more information.

        Args:
            configuration: Robot configuration :math:`q`.
            dt: Integration timestep in [s].

        Returns:
            Pair :math:`(G, h)` representing the inequality constraint as
            :math:`G \Delta q \leq h`, or ``None`` if there is no limit.
        """
        del dt  # Unused.
        if self.projection_matrix is None and len(self._ball_qposadr) == 0:
            return Constraint()

        G_list: list[np.ndarray] = []
        h_list: list[np.ndarray] = []
        q = configuration.q

        if self.projection_matrix is not None:
            # Upper.
            delta_q_max = np.empty((self.model.nv,))
            mujoco.mj_differentiatePos(
                m=self.model,
                qvel=delta_q_max,
                dt=1.0,
                qpos1=q,
                qpos2=self.upper,
            )

            # Lower.
            delta_q_min = np.empty((self.model.nv,))
            mujoco.mj_differentiatePos(
                m=self.model,
                qvel=delta_q_min,
                dt=1.0,
                # NOTE: mujoco.mj_differentiatePos does `qpos2 - qpos1` so notice the
                # order swap here compared to above.
                qpos1=self.lower,
                qpos2=q,
            )

            p_min = self.gain * delta_q_min[self.indices]
            p_max = self.gain * delta_q_max[self.indices]
            G_list.append(self.projection_matrix)
            G_list.append(-self.projection_matrix)
            h_list.append(p_max)
            h_list.append(p_min)

        # Ball joints: a^T dq <= gain * (max_angle - angle) along the rotation axis.
        phi = self._phi
        for i, (padr, dadr, max_angle) in enumerate(
            zip(self._ball_qposadr, self._ball_dofadr, self._ball_max_angle)
        ):
            mujoco.mju_quat2Vel(phi, q[padr : padr + 4], 1.0)
            angle = float(np.linalg.norm(phi))
            axis = self._ball_G[i, dadr : dadr + 3]
            # The axis is undefined at the identity; leave the row zero.
            if angle > 1e-9:
                axis[:] = phi
                axis /= angle
            else:
                axis[:] = 0.0
            self._ball_h[i] = self.gain * (max_angle - angle)
        if len(self._ball_h) > 0:
            G_list.append(self._ball_G)
            h_list.append(self._ball_h)

        return Constraint(G=np.vstack(G_list), h=np.hstack(h_list))
