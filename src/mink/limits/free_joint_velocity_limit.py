"""Free-joint (floating base) velocity limit."""

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..exceptions import LimitDefinitionError
from .limit import Constraint, Limit


class FreeJointVelocityLimit(Limit):
    r"""Velocity limit on a free joint (floating base).

    :class:`VelocityLimit` skips free joints; this bounds their six velocity DOFs
    as a linear and an angular part, both expressed in the base body frame.

    MuJoCo stores the free-joint linear velocity in the world frame and the
    angular velocity in the body frame. The angular bound is thus applied
    directly, while the linear bound is rotated into the body frame by the base
    orientation :math:`R`, giving :math:`|R^\top v| \leq v_{\max}`.

    Attributes:
        linear_max: Linear velocity bound in [m]/[s], or None. Shape (3,).
        angular_max: Angular velocity bound in [rad]/[s], or None. Shape (3,).
    """

    linear_max: np.ndarray | None
    angular_max: np.ndarray | None

    def __init__(
        self,
        model: mujoco.MjModel,
        max_linear_velocity: npt.ArrayLike | None = None,
        max_angular_velocity: npt.ArrayLike | None = None,
        joint_name: str | None = None,
    ):
        """Initialize the free-joint velocity limit.

        Args:
            model: MuJoCo model.
            max_linear_velocity: Linear velocity bound in [m]/[s], scalar or
                length-3, in the base frame. None leaves the linear part free.
            max_angular_velocity: Angular velocity bound in [rad]/[s], scalar or
                length-3, in the base frame. None leaves the angular part free.
            joint_name: Free joint to limit. If None, the model's sole free joint
                is used; an error is raised for zero or multiple free joints.
        """
        jid = self._resolve_free_joint(model, joint_name)
        self.base_body_id = int(model.jnt_bodyid[jid])
        dofadr = int(model.jnt_dofadr[jid])

        limits: list[np.ndarray | None] = []
        for value, name in (
            (max_linear_velocity, "max_linear_velocity"),
            (max_angular_velocity, "max_angular_velocity"),
        ):
            if value is None:
                limits.append(None)
                continue
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 0:
                arr = np.full(3, arr)
            if arr.shape != (3,):
                raise LimitDefinitionError(
                    f"{name} must be a scalar or length 3. Got shape {arr.shape}"
                )
            arr.setflags(write=False)
            limits.append(arr)
        self.linear_max, self.angular_max = limits
        if self.linear_max is None and self.angular_max is None:
            raise LimitDefinitionError(
                "At least one of max_linear_velocity or max_angular_velocity "
                "must be set"
            )

        # Linear DOFs are world-frame (rotated per step); angular DOFs are already
        # body-frame, so their block is static.
        self._lin_dofs = dofadr + np.arange(3)
        ang_dofs = dofadr + np.arange(3, 6)
        self._G_ang = (
            np.eye(model.nv)[ang_dofs] if self.angular_max is not None else None
        )
        self._nv = model.nv

    @staticmethod
    def _resolve_free_joint(model: mujoco.MjModel, joint_name: str | None) -> int:
        if joint_name is not None:
            jid = model.joint(joint_name).id
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                raise LimitDefinitionError(f"Joint {joint_name} is not a free joint")
            return jid
        free_jids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free_jids) == 0:
            raise LimitDefinitionError("Model has no free joint")
        if len(free_jids) > 1:
            names = [model.joint(int(j)).name for j in free_jids]
            raise LimitDefinitionError(
                f"Model has multiple free joints {names}; pass joint_name"
            )
        return int(free_jids[0])

    def compute_qp_inequalities(
        self, configuration: Configuration, dt: float
    ) -> Constraint:
        r"""Compute the free-joint velocity limit as :math:`G \Delta q \leq h`."""
        blocks: list[np.ndarray] = []
        bounds: list[np.ndarray] = []
        if self.linear_max is not None:
            R = configuration.data.xmat[self.base_body_id].reshape(3, 3)
            G_lin = np.zeros((3, self._nv))
            G_lin[:, self._lin_dofs] = R.T  # base-from-world rotation.
            blocks.append(G_lin)
            bounds.append(self.linear_max)
        if self._G_ang is not None:
            blocks.append(self._G_ang)
            assert self.angular_max is not None
            bounds.append(self.angular_max)
        G_active = np.vstack(blocks)
        h_active = dt * np.hstack(bounds)
        G = np.vstack([G_active, -G_active])
        h = np.hstack([h_active, h_active])
        return Constraint(G=G, h=h)
