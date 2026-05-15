"""Configuration space of a robot model.

The :class:`Configuration` class encapsulates a MuJoCo
`model <https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjmodel>`__
and `data <https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjdata>`__,
offering easy access to frame transforms and frame Jacobians. A frame refers to a coordinate
system that can be attached to various parts of the robot, such as a body, geom, or site.
"""

import logging
import os

import mujoco
import numpy as np

from . import constants as consts
from . import exceptions
from .lie import SE3, SO3

try:
    if os.environ.get("MINK_DISABLE_NATIVE", ""):
        raise ImportError
    from .lie import _lie_ops_c as _native  # noqa: PLC0415
except ImportError:
    _native = None  # type: ignore[assignment]


class Configuration:
    """Encapsulates a model and data for convenient access to kinematic quantities.

    This class provides methods to access and update the kinematic quantities of a
    robot model, such as frame transforms and Jacobians. It performs forward kinematics
    at every time step, ensuring up-to-date information about the robot's state.

    Key functionalities include:

    * Running forward kinematics to update the state.
    * Checking configuration limits.
    * Computing Jacobians for different frames.
    * Computing the joint-space inertia matrix.
    * Retrieving frame transforms relative to the world frame.
    * Integrating velocities to update configurations.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        q: np.ndarray | None = None,
    ):
        """Constructor.

        Args:
            model: Mujoco model.
            q: Configuration to initialize from. If None, the configuration is
                initialized to the default configuration `qpos0`.
        """
        self.model = model
        self.data = mujoco.MjData(model)
        self._logger = logging.getLogger(__package__)

        # Precompute limited joint indices for vectorized check_limits.
        limited = model.jnt_limited.astype(bool)
        limited &= model.jnt_type != mujoco.mjtJoint.mjJNT_FREE
        self._limited_jnt_ids = np.where(limited)[0]
        self._limited_qposadr = model.jnt_qposadr[self._limited_jnt_ids]
        self._limited_range = model.jnt_range[self._limited_jnt_ids]

        # Cached identity matrix for QP assembly.
        self._eye_nv = np.eye(model.nv)

        self.update(q=q)

    def update(self, q: np.ndarray | None = None) -> None:
        """Run forward kinematics.

        Args:
            q: Optional configuration vector to override internal `data.qpos` with.
        """
        if q is not None:
            self.data.qpos = q
        # The minimal function call required to get updated frame transforms is
        # mj_kinematics. An extra call to mj_comPos is required for updated Jacobians.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        if self.model.neq > 0:
            mujoco.mj_makeConstraint(self.model, self.data)

    def update_from_keyframe(self, key_name: str) -> None:
        """Update the configuration from a keyframe.

        Args:
            key_name: The name of the keyframe.
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

        Raises:
            NotWithinConfigurationLimits: If the current configuration is outside
                the joint limits.
        """
        if len(self._limited_jnt_ids) == 0:
            return
        qvals = self.data.qpos[self._limited_qposadr]
        violations = (qvals < self._limited_range[:, 0] - tol) | (
            qvals > self._limited_range[:, 1] + tol
        )
        if not violations.any():
            return
        if safety_break:
            idx = int(np.argmax(violations))
            jnt = int(self._limited_jnt_ids[idx])
            raise exceptions.NotWithinConfigurationLimits(
                joint_id=jnt,
                value=int(qvals[idx]),
                lower=self._limited_range[idx, 0],
                upper=self._limited_range[idx, 1],
                model=self.model,
            )
        for idx in np.where(violations)[0]:
            jnt = int(self._limited_jnt_ids[idx])
            qval = qvals[idx]
            qmin = self._limited_range[idx, 0]
            qmax = self._limited_range[idx, 1]
            self._logger.debug(
                f"Value {qval:.2f} at joint {jnt} is outside of its limits: "
                f"[{qmin:.2f}, {qmax:.2f}]"
            )

    def get_frame_jacobian(self, frame_name: str, frame_type: str) -> np.ndarray:
        r"""Compute the Jacobian matrix of a frame velocity.

        Denoting our frame by :math:`B` and the world frame by :math:`W`, the
        Jacobian matrix :math:`{}_B J_{WB}` is related to the body velocity
        :math:`{}_B v_{WB}` by:

        .. math::

            {}_B v_{WB} = {}_B J_{WB} \dot{q}

        Args:
            frame_name: Name of the frame in the MJCF.
            frame_type: Type of frame. Can be a geom, a body or a site.

        Returns:
            Jacobian :math:`{}_B J_{WB}` of the frame.
        """
        frame_id = self._resolve_frame_id(frame_name, frame_type)

        jac = np.empty((6, self.model.nv))
        jac_func = consts.FRAME_TO_JAC_FUNC[frame_type]
        jac_func(self.model, self.data, jac[:3], jac[3:], frame_id)

        # MuJoCo jacobians have a frame of reference centered at the local frame but
        # aligned with the world frame. To obtain a jacobian expressed in the local
        # frame, aka body jacobian, we need to left-multiply by A[T_fw].
        xmat = getattr(self.data, consts.FRAME_TO_XMAT_ATTR[frame_type])[frame_id]
        if _native is not None:
            A_fw = _native.se3_rotation_adjoint_from_xmat(xmat)
        else:
            R_wf = SO3.from_matrix(xmat.reshape(3, 3))
            A_fw = SE3.from_rotation(R_wf.inverse()).adjoint()
        jac = A_fw @ jac

        return jac

    def _resolve_frame_id(self, frame_name: str, frame_type: str) -> int:
        """Validate frame type and resolve name to ID."""
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
        return frame_id

    def _get_transform_frame_to_world_wxyz_xyz(
        self, frame_name: str, frame_type: str
    ) -> np.ndarray:
        """Return the raw wxyz_xyz[7] array for a frame pose. Internal use."""
        frame_id = self._resolve_frame_id(frame_name, frame_type)
        xpos = getattr(self.data, consts.FRAME_TO_POS_ATTR[frame_type])[frame_id]
        xmat = getattr(self.data, consts.FRAME_TO_XMAT_ATTR[frame_type])[frame_id]
        if _native is not None:
            return _native.xmat_xpos_to_wxyz_xyz(xmat, xpos)
        return SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(xmat.reshape(3, 3)),
            translation=xpos,
        ).wxyz_xyz

    def get_transform_frame_to_world(self, frame_name: str, frame_type: str) -> SE3:
        """Get the pose of a frame at the current configuration.

        Args:
            frame_name: Name of the frame in the MJCF.
            frame_type: Type of frame. Can be a geom, a body or a site.

        Returns:
            The pose of the frame in the world frame.
        """
        return SE3(
            wxyz_xyz=self._get_transform_frame_to_world_wxyz_xyz(frame_name, frame_type)
        )

    def _get_transform_wxyz_xyz(
        self,
        source_name: str,
        source_type: str,
        dest_name: str,
        dest_type: str,
    ) -> np.ndarray:
        """Return the raw wxyz_xyz[7] for a relative transform. Internal use."""
        source = self._get_transform_frame_to_world_wxyz_xyz(source_name, source_type)
        dest = self._get_transform_frame_to_world_wxyz_xyz(dest_name, dest_type)
        if _native is not None:
            return _native.se3_inverse_multiply(dest, source)
        dest_se3 = SE3(wxyz_xyz=dest)
        source_se3 = SE3(wxyz_xyz=source)
        return (dest_se3.inverse() @ source_se3).wxyz_xyz

    def get_transform(
        self,
        source_name: str,
        source_type: str,
        dest_name: str,
        dest_type: str,
    ) -> SE3:
        """Get the pose of a frame with respect to another frame at the current
        configuration.

        Args:
            source_name: Name of the frame in the MJCF.
            source_type: Source type of frame. Can be a geom, a body or a site.
            dest_name: Name of the frame to get the pose in.
            dest_type: Dest type of frame. Can be a geom, a body or a site.

        Returns:
            The pose of `source_name` in `dest_name`.
        """
        return SE3(
            wxyz_xyz=self._get_transform_wxyz_xyz(
                source_name, source_type, dest_name, dest_type
            )
        )

    def integrate(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        """Integrate a velocity starting from the current configuration.

        Args:
            velocity: The velocity in tangent space.
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
            velocity: The velocity in tangent space.
            dt: Integration duration in [s].
        """
        mujoco.mj_integratePos(self.model, self.data.qpos, velocity, dt)
        self.update()

    def get_inertia_matrix(self) -> np.ndarray:
        r"""Return the joint-space inertia matrix at the current configuration.

        Returns:
            The joint-space inertia matrix :math:`M(\mathbf{q})`.
        """
        # Run the composite rigid body inertia (CRB) algorithm to populate the joint
        # space inertia matrix data.M.
        mujoco.mj_makeM(self.model, self.data)
        # data.M is stored in a lower-triangular implicitly-symmetric CSR format and
        # can be converted to a dense symmetric matrix via mujoco.mju_sym2dense.
        M = np.empty((self.nv, self.nv), dtype=np.float64)
        mujoco.mju_sym2dense(
            M,
            self.data.M,
            self.model.M_rownnz,
            self.model.M_rowadr,
            self.model.M_colind,
        )
        return M

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
