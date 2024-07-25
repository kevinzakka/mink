from typing import Optional

import mujoco
import numpy as np

from . import constants as consts
from . import exceptions
from .lie import SE3, SO3


class Configuration:
    """A struct that provides convenient access to kinematic quantities such as frame
    transforms and frame jacobians. Frames can be defined at bodies, geoms or sites.

    The `update` function ensures the proper forward kinematics functions have been
    called, namely:

    - mujoco.mj_kinematics(model, data)
    - mujoco.mj_comPos(model, data)
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        q: Optional[np.ndarray] = None,
    ):
        """Constructor.

        Args:
            model: An instance of MjModel.
            q: Optional configuration to initialize from. If None, the configuration
                is initialized to the reference configuration `qpos0`.
        """
        self.model = model
        self.data = mujoco.MjData(model)
        self.update(q=q)

    def update(self, q: Optional[np.ndarray] = None) -> None:
        """Run forward kinematics.

        The minimal function call required to get updated frame transforms (aka forward
        kinematics) is `mj_kinematics`. An extra call to `mj_comPos` is needed for
        updated Jacobians.

        Args:`
            q: Optional configuration vector to override internal data.qpos with.
        """
        if q is not None:
            self.data.qpos = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def update_from_keyframe(self, key_name: str) -> None:
        """Update the configuration from a keyframe.

        Args:
            key_name: The name of the keyframe.

        Raises:
            ValueError: if no key named `key` was found in the model.
        """
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        if key_id == -1:
            raise exceptions.InvalidKeyframe(key_name, self.model)
        self.update(q=self.model.key_qpos[key_id])

    def check_limits(self, tol: float = 1e-6, safety_break: bool = True) -> None:
        """Check that the current configuration is within bounds.

        Args:
            tol: Tolerance in [rad].
            safety_break: If True, stop execution and raise an exception if the current
                configuration is outside limits. If False, print a warning and continue
                execution.
        """
        for jnt in range(self.model.njnt):
            jnt_type = self.model.jnt_type[jnt]
            if (
                jnt_type == mujoco.mjtJoint.mjJNT_FREE
                or not self.model.jnt_limited[jnt]
            ):
                continue
            padr = self.model.jnt_qposadr[jnt]
            qval = self.q[padr]
            qmin = self.model.jnt_range[jnt, 0]
            qmax = self.model.jnt_range[jnt, 1]
            if qval < qmin - tol or qval > qmax + tol:
                if safety_break:
                    raise exceptions.NotWithinConfigurationLimits(
                        joint_id=jnt,
                        value=qval,
                        lower=qmin,
                        upper=qmax,
                        model=self.model,
                    )
                else:
                    print(
                        f"Value {qval} at index {jnt} is out of limits: "
                        f"[{qmin}, {qmax}]"
                    )

    def get_frame_jacobian(self, frame_name: str, frame_type: str) -> np.ndarray:
        """Compute the Jacobian matrix of a frame velocity.

        Denoting our frame by B and the world frame by W, the Jacobian matrix B_J_WB
        is related to the body velocity B_v_WB by:

            B_v_WB = B_J_WB q_dot

        Args:
            frame_name: Name of the frame in the MJCF.
            frame_type: Type of frame. Can be a geom, a body or a site.

        Returns:
            Jacobian B_J_WB of the frame.
        """
        if frame_type not in consts.SUPPORTED_FRAMES:
            raise exceptions.UnsupportedFrame(frame_type, consts.SUPPORTED_FRAMES)

        frame_id = mujoco.mj_name2id(
            self.model, consts.FRAME_TO_ENUM[frame_type], frame_name
        )
        if frame_id == -1:
            raise exceptions.InvalidFrame(
                frame_name=frame_name,
                frame_type=frame_type,
                model=self.model,
            )

        jac = np.empty((6, self.model.nv))
        jac_func = consts.FRAME_TO_JAC_FUNC[frame_type]
        jac_func(self.model, self.data, jac[:3], jac[3:], frame_id)

        # MuJoCo jacobians have a frame of reference centered at the local frame but
        # aligned with the world frame. To obtain a jacobian expressed in the local
        # frame, aka body jacobian, we need to left-multiply by A[T_fw].
        xmat = getattr(self.data, consts.FRAME_TO_XMAT_ATTR[frame_type])[frame_id]
        R_wf = SO3.from_matrix(xmat.reshape(3, 3))
        A_fw = SE3.from_rotation(R_wf.inverse()).adjoint()
        jac = A_fw @ jac

        return jac

    def get_transform_frame_to_world(self, frame_name: str, frame_type: str) -> SE3:
        """Get the pose of a frame in the current configuration.

        Denoting our frame by B and the world frame by W, this function returns T_WB.

        Args:
            frame_name: Name of the frame in the MJCF.
            frame_type: Type of frame. Can be a geom, a body or a site.

        Returns:
            The pose of the frame in the world frame.
        """
        if frame_type not in consts.SUPPORTED_FRAMES:
            raise exceptions.UnsupportedFrame(frame_type, consts.SUPPORTED_FRAMES)

        frame_id = mujoco.mj_name2id(
            self.model, consts.FRAME_TO_ENUM[frame_type], frame_name
        )
        if frame_id == -1:
            raise exceptions.InvalidFrame(
                frame_name=frame_name,
                frame_type=frame_type,
                model=self.model,
            )

        xpos = getattr(self.data, consts.FRAME_TO_POS_ATTR[frame_type])[frame_id]
        xmat = getattr(self.data, consts.FRAME_TO_XMAT_ATTR[frame_type])[frame_id]
        return SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(xmat.reshape(3, 3)),
            translation=xpos,
        )

    def integrate(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        """Integrate a velocity starting from the current configuration.

        Args:
            velocity: The velocity.
            dt: Integration duration in [s].

        Returns:
            The new configuration after integration.
        """
        q = self.data.qpos.copy()
        mujoco.mj_integratePos(self.model, q, velocity, dt)
        return q

    def integrate_inplace(self, velocity: np.ndarray, dt: float) -> None:
        """Integrate a velocity and update the current configuration inplace.

        Args:
            velocity: The velocity.
            dt: Integration duration in [s].
        """
        mujoco.mj_integratePos(self.model, self.data.qpos, velocity, dt)
        self.update()

    # Aliases.

    @property
    def q(self) -> np.ndarray:
        """The current configuration vector."""
        return self.data.qpos.copy()

    @property
    def nv(self) -> int:
        """The dimension of the tangent space."""
        return self.model.nv

    @property
    def nq(self) -> int:
        """The dimension of the configuration space."""
        return self.model.nq
