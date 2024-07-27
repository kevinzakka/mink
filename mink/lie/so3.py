from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import mujoco
import numpy as np

from .base import MatrixLieGroup
from .utils import get_epsilon, skew

_IDENTITIY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
_INVERT_QUAT_SIGN = np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float64)


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
        wxyz = np.zeros(SO3.parameters_dim, dtype=np.float64)
        mujoco.mju_mat2Quat(wxyz, matrix.ravel())
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
        mat = np.zeros(9, dtype=np.float64)
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

    # Paragraph above Appendix B.A.
    def inverse(self) -> SO3:
        return SO3(wxyz=self.wxyz * _INVERT_QUAT_SIGN)

    def normalize(self) -> SO3:
        return SO3(wxyz=self.wxyz / np.linalg.norm(self.wxyz))

    # Eq. 136.
    def apply(self, target: np.ndarray) -> np.ndarray:
        assert target.shape == (SO3.space_dim,)
        padded_target = np.concatenate([np.zeros(1, dtype=np.float64), target])
        return (self @ SO3(wxyz=padded_target) @ self.inverse()).wxyz[1:]

    def multiply(self, other: SO3) -> SO3:
        res = np.empty(self.parameters_dim, dtype=np.float64)
        mujoco.mju_mulQuat(res, self.wxyz, other.wxyz)
        return SO3(wxyz=res)

    ##
    #
    ##

    # Eq. 132.
    @classmethod
    def exp(cls, tangent: np.ndarray) -> SO3:
        assert tangent.shape == (SO3.tangent_dim,)
        theta_squared = tangent @ tangent
        theta_pow_4 = theta_squared * theta_squared
        use_taylor = theta_squared < get_epsilon(tangent.dtype)
        safe_theta = 1.0 if use_taylor else np.sqrt(theta_squared)
        safe_half_theta = 0.5 * safe_theta
        if use_taylor:
            real = 1.0 - theta_squared / 8.0 + theta_pow_4 / 384.0
            imaginary = 0.5 - theta_squared / 48.0 + theta_pow_4 / 3840.0
        else:
            real = np.cos(safe_half_theta)
            imaginary = np.sin(safe_half_theta) / safe_theta
        wxyz = np.concatenate([np.array([real]), imaginary * tangent])
        return SO3(wxyz=wxyz)

    # Eq. 133.
    def log(self) -> np.ndarray:
        w = self.wxyz[0]
        norm_sq = self.wxyz[1:] @ self.wxyz[1:]
        use_taylor = norm_sq < get_epsilon(norm_sq.dtype)
        norm_safe = 1.0 if use_taylor else np.sqrt(norm_sq)
        w_safe = w if use_taylor else 1.0
        atan_n_over_w = np.arctan2(-norm_safe if w < 0 else norm_safe, abs(w))
        if use_taylor:
            atan_factor = 2.0 / w_safe - 2.0 / 3.0 * norm_sq / w_safe**3
        else:
            if abs(w) < get_epsilon(w.dtype):
                scl = 1.0 if w > 0.0 else -1.0
                atan_factor = scl * np.pi / norm_safe
            else:
                atan_factor = 2.0 * atan_n_over_w / norm_safe
        return atan_factor * self.wxyz[1:]

    # Eq. 139.
    def adjoint(self) -> np.ndarray:
        return self.as_matrix()

    # Jacobians.

    # Eqn. 145, 174.
    @classmethod
    def ljac(cls, other: np.ndarray) -> np.ndarray:
        theta = np.sqrt(other @ other)
        use_taylor = theta < get_epsilon(theta.dtype)
        if use_taylor:
            t2 = theta**2
            A = (1.0 / 2.0) * (1.0 - t2 / 12.0 * (1.0 - t2 / 30.0 * (1.0 - t2 / 56.0)))
            B = (1.0 / 6.0) * (1.0 - t2 / 20.0 * (1.0 - t2 / 42.0 * (1.0 - t2 / 72.0)))
        else:
            A = (1 - np.cos(theta)) / (theta**2)
            B = (theta - np.sin(theta)) / (theta**3)
        skew_other = skew(other)
        return np.eye(3) + A * skew_other + B * (skew_other @ skew_other)

    @classmethod
    def ljacinv(cls, other: np.ndarray) -> np.ndarray:
        theta = np.sqrt(other @ other)
        use_taylor = theta < get_epsilon(theta.dtype)
        if use_taylor:
            t2 = theta**2
            A = (1.0 / 12.0) * (1.0 + t2 / 60.0 * (1.0 + t2 / 42.0 * (1.0 + t2 / 40.0)))
        else:
            A = (1.0 / theta**2) * (
                1.0 - (theta * np.sin(theta) / (2.0 * (1.0 - np.cos(theta))))
            )
        skew_other = skew(other)
        return np.eye(3) - 0.5 * skew_other + A * (skew_other @ skew_other)
