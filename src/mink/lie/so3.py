from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import mujoco
import numpy as np

from .base import MatrixLieGroup

_IDENTITIY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
# Squared-angle cutoff: below this, the closed forms lose precision to
# subtractive cancellation while the Taylor series remain effectively exact.
_JACOBIAN_TAYLOR_THRESHOLD_SQUARED = 1e-2


class RollPitchYaw(NamedTuple):
    """Struct containing roll, pitch, and yaw Euler angles."""

    roll: float
    pitch: float
    yaw: float


@dataclass(frozen=True)
class SO3(MatrixLieGroup):
    """Special orthogonal group for 3D rotations.

    Internal parameterization is (qw, qx, qy, qz). Tangent parameterization is
    (omega_x, omega_y, omega_z).
    """

    wxyz: np.ndarray
    matrix_dim: int = 3
    parameters_dim: int = 4
    tangent_dim: int = 3
    space_dim: int = 3

    def __post_init__(self) -> None:
        if self.wxyz.shape != (self.parameters_dim,):
            raise ValueError(
                f"Expeced wxyz to be a length 4 vector but got {self.wxyz.shape[0]}."
            )

    def __repr__(self) -> str:
        wxyz = np.round(self.wxyz, 5)
        return f"{self.__class__.__name__}(wxyz={wxyz})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SO3):
            return NotImplemented
        return np.array_equal(self.wxyz, other.wxyz)

    def __hash__(self) -> int:
        return hash(self.wxyz.tobytes())

    def parameters(self) -> np.ndarray:
        return self.wxyz

    def copy(self) -> SO3:
        return SO3(wxyz=self.wxyz.copy())

    @classmethod
    def from_x_radians(cls, theta: float) -> SO3:
        return SO3.exp(np.array([theta, 0.0, 0.0], dtype=np.float64))

    @classmethod
    def from_y_radians(cls, theta: float) -> SO3:
        return SO3.exp(np.array([0.0, theta, 0.0], dtype=np.float64))

    @classmethod
    def from_z_radians(cls, theta: float) -> SO3:
        return SO3.exp(np.array([0.0, 0.0, theta], dtype=np.float64))

    @classmethod
    def from_rpy_radians(
        cls,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> SO3:
        return (
            SO3.from_z_radians(yaw)
            @ SO3.from_y_radians(pitch)
            @ SO3.from_x_radians(roll)
        )

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> SO3:
        assert matrix.shape == (SO3.matrix_dim, SO3.matrix_dim)
        wxyz = np.empty(SO3.parameters_dim, dtype=np.float64)
        mujoco.mju_mat2Quat(wxyz, matrix.ravel())
        # NOTE mju_mat2Quat normalizes the quaternion.
        return SO3(wxyz=wxyz)

    @classmethod
    def identity(cls) -> SO3:
        return SO3(wxyz=_IDENTITIY_WXYZ)

    @classmethod
    def sample_uniform(cls) -> SO3:
        # Ref: https://lavalle.pl/planning/node198.html
        u1, u2, u3 = np.random.uniform(
            low=np.zeros(shape=(3,)),
            high=np.array([1.0, 2.0 * np.pi, 2.0 * np.pi]),
        )
        a = np.sqrt(1.0 - u1)
        b = np.sqrt(u1)
        wxyz = np.array(
            [
                a * np.sin(u2),
                a * np.cos(u2),
                b * np.sin(u3),
                b * np.cos(u3),
            ],
            dtype=np.float64,
        )
        return SO3(wxyz=wxyz)

    # Eq. 138.
    def as_matrix(self) -> np.ndarray:
        mat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, self.wxyz)
        return mat.reshape(3, 3)

    def compute_roll_radians(self) -> float:
        q0, q1, q2, q3 = self.wxyz
        return np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))

    def compute_pitch_radians(self) -> float:
        q0, q1, q2, q3 = self.wxyz
        return np.arcsin(2 * (q0 * q2 - q3 * q1))

    def compute_yaw_radians(self) -> float:
        q0, q1, q2, q3 = self.wxyz
        return np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))

    def as_rpy_radians(self) -> RollPitchYaw:
        return RollPitchYaw(
            roll=self.compute_roll_radians(),
            pitch=self.compute_pitch_radians(),
            yaw=self.compute_yaw_radians(),
        )

    def inverse(self) -> SO3:
        conjugate_wxyz = np.empty(4)
        mujoco.mju_negQuat(conjugate_wxyz, self.wxyz)
        return SO3(wxyz=conjugate_wxyz)

    def normalize(self) -> SO3:
        normalized_wxyz = np.array(self.wxyz)
        mujoco.mju_normalize4(normalized_wxyz)
        return SO3(wxyz=normalized_wxyz)

    # Eq. 136.
    def apply(self, target: np.ndarray) -> np.ndarray:
        assert target.shape == (SO3.space_dim,)
        rotated_target = np.empty(SO3.space_dim, dtype=np.float64)
        mujoco.mju_rotVecQuat(rotated_target, target, self.wxyz)
        return rotated_target

    def multiply(self, other: SO3) -> SO3:  # type: ignore[override]
        res = np.empty(self.parameters_dim, dtype=np.float64)
        mujoco.mju_mulQuat(res, self.wxyz, other.wxyz)
        return SO3(wxyz=res)

    # Eq. 132.
    @classmethod
    def exp(cls, tangent: np.ndarray) -> SO3:
        theta = np.linalg.norm(tangent)
        half_theta = 0.5 * theta
        wxyz = np.empty(4, dtype=np.float64)
        wxyz[0] = np.cos(half_theta)
        if theta == 0.0:
            wxyz[1:] = 0.0
        else:
            wxyz[1:] = (np.sin(half_theta) / theta) * tangent
        return SO3(wxyz=wxyz)

    # Eq. 133.
    def log(self) -> np.ndarray:
        q = np.array(self.wxyz)
        if q[0] < 0.0:
            q = -q
        w, v = q[0], q[1:]
        norm = np.linalg.norm(v)
        if norm == 0.0:
            return np.zeros_like(v)
        return (2 * np.arctan2(norm, w) / norm) * v

    # Eq. 139.
    def adjoint(self) -> np.ndarray:
        return self.as_matrix()

    # Jacobians.

    # Eqn. 145, 174.
    @classmethod
    def ljac(cls, other: np.ndarray) -> np.ndarray:
        theta = np.float64(mujoco.mju_norm3(other))
        t2 = theta * theta
        if t2 < _JACOBIAN_TAYLOR_THRESHOLD_SQUARED:
            alpha = (1.0 / 2.0) * (
                1.0 - t2 / 12.0 * (1.0 - t2 / 30.0 * (1.0 - t2 / 56.0))
            )
            beta = (1.0 / 6.0) * (
                1.0 - t2 / 20.0 * (1.0 - t2 / 42.0 * (1.0 - t2 / 72.0))
            )
        else:
            t3 = t2 * theta
            alpha = (1 - np.cos(theta)) / t2
            beta = (theta - np.sin(theta)) / t3
        # ljac = eye(3) + alpha * skew_other + beta * (skew_other @ skew_other)
        ljac = np.empty((3, 3))
        # skew_other @ skew_other == outer(other) - inner(other) * eye(3)
        mujoco.mju_mulMatMat(ljac, other.reshape(3, 1), other.reshape(1, 3))
        inner_product = mujoco.mju_dot3(other, other)
        ljac[0, 0] -= inner_product
        ljac[1, 1] -= inner_product
        ljac[2, 2] -= inner_product
        ljac *= beta
        # + alpha * skew_other
        alpha_vec = alpha * other
        ljac[0, 1] += -alpha_vec[2]
        ljac[0, 2] += alpha_vec[1]
        ljac[1, 0] += alpha_vec[2]
        ljac[1, 2] += -alpha_vec[0]
        ljac[2, 0] += -alpha_vec[1]
        ljac[2, 1] += alpha_vec[0]
        # + eye(3)
        ljac[0, 0] += 1.0
        ljac[1, 1] += 1.0
        ljac[2, 2] += 1.0
        return ljac

    @classmethod
    def ljacinv(cls, other: np.ndarray) -> np.ndarray:
        theta = np.float64(mujoco.mju_norm3(other))
        t2 = theta * theta
        if t2 < _JACOBIAN_TAYLOR_THRESHOLD_SQUARED:
            beta = (1.0 / 12.0) * (
                1.0 + t2 / 60.0 * (1.0 + t2 / 42.0 * (1.0 + t2 / 40.0))
            )
        else:
            half_theta = 0.5 * theta
            beta = (1.0 - half_theta / np.tan(half_theta)) / t2
        # ljacinv = eye(3) - 0.5 * skew_other + beta * (skew_other @ skew_other)
        ljacinv = np.empty((3, 3))
        # skew_other @ skew_other == outer(other) - inner(other) * eye(3)
        mujoco.mju_mulMatMat(ljacinv, other.reshape(3, 1), other.reshape(1, 3))
        inner_product = mujoco.mju_dot3(other, other)
        ljacinv[0, 0] -= inner_product
        ljacinv[1, 1] -= inner_product
        ljacinv[2, 2] -= inner_product
        ljacinv *= beta
        # - 0.5 * skew_other
        alpha_vec = -0.5 * other
        ljacinv[0, 1] += -alpha_vec[2]
        ljacinv[0, 2] += alpha_vec[1]
        ljacinv[1, 0] += alpha_vec[2]
        ljacinv[1, 2] += -alpha_vec[0]
        ljacinv[2, 0] += -alpha_vec[1]
        ljacinv[2, 1] += alpha_vec[0]
        # + eye(3)
        ljacinv[0, 0] += 1.0
        ljacinv[1, 1] += 1.0
        ljacinv[2, 2] += 1.0
        return ljacinv

    def clamp(
        self,
        roll_radians: tuple[float, float] = (-np.inf, np.inf),
        pitch_radians: tuple[float, float] = (-np.inf, np.inf),
        yaw_radians: tuple[float, float] = (-np.inf, np.inf),
    ) -> SO3:
        """Clamp a SO3 within RPY limits.

        Args:
            roll_radians: The (lower, upper) limits for the roll.
            pitch_radians: The (lower, upper) limits for the pitch.
            yaw_radians: The (lower, upper) limits for the yaw.

        Returns:
            A SO3 within the RPY limits.
        """
        return SO3.from_rpy_radians(
            roll=np.clip(self.compute_roll_radians(), *roll_radians),
            pitch=np.clip(self.compute_pitch_radians(), *pitch_radians),
            yaw=np.clip(self.compute_yaw_radians(), *yaw_radians),
        )
