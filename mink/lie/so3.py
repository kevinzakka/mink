from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .utils import get_epsilon, skew

_IDENTITIY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
_INVERT_QUAT_SIGN = np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float64)


@dataclass(frozen=True)
class RollPitchYaw:
    roll: float
    pitch: float
    yaw: float


@dataclass(frozen=True)
class SO3:
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

    def copy(self) -> SO3:
        return SO3(wxyz=self.wxyz.copy())

    @staticmethod
    def from_x_radians(theta: float) -> SO3:
        return SO3.exp(np.array([theta, 0.0, 0.0], dtype=np.float64))

    @staticmethod
    def from_y_radians(theta: float) -> SO3:
        return SO3.exp(np.array([0.0, theta, 0.0], dtype=np.float64))

    @staticmethod
    def from_z_radians(theta: float) -> SO3:
        return SO3.exp(np.array([0.0, 0.0, theta], dtype=np.float64))

    @staticmethod
    def from_rpy_radians(
        roll: float,
        pitch: float,
        yaw: float,
    ) -> SO3:
        return (
            SO3.from_z_radians(yaw)
            @ SO3.from_y_radians(pitch)
            @ SO3.from_x_radians(roll)
        )

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> SO3:
        assert matrix.shape == (SO3.matrix_dim, SO3.matrix_dim)
        wxyz = np.zeros(SO3.parameters_dim, dtype=np.float64)
        mujoco.mju_mat2Quat(wxyz, matrix.ravel())
        return SO3(wxyz=wxyz)

    @staticmethod
    def identity() -> SO3:
        return SO3(wxyz=_IDENTITIY_WXYZ)

    @staticmethod
    def sample_uniform() -> SO3:
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

    @staticmethod
    def exp(tangent: np.ndarray) -> SO3:
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

    def adjoint(self) -> np.ndarray:
        return self.as_matrix()

    def inverse(self) -> SO3:
        return SO3(wxyz=self.wxyz * _INVERT_QUAT_SIGN)

    def normalize(self) -> SO3:
        return SO3(wxyz=self.wxyz / np.linalg.norm(self.wxyz))

    def apply(self, target: np.ndarray) -> np.ndarray:
        assert target.shape == (SO3.space_dim,)
        padded_target = np.concatenate([np.zeros(1, dtype=np.float64), target])
        return (self @ SO3(wxyz=padded_target) @ self.inverse()).wxyz[1:]

    def multiply(self, other: SO3) -> SO3:
        w0, x0, y0, z0 = self.wxyz
        w1, x1, y1, z1 = other.wxyz
        return SO3(
            wxyz=np.array(
                [
                    -x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1,
                    x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1,
                    -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1,
                    x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1,
                ],
                dtype=np.float64,
            )
        )

    def __matmul__(self, other: SO3 | np.ndarray) -> SO3 | np.ndarray:
        if isinstance(other, np.ndarray):
            return self.apply(target=other)
        elif isinstance(other, SO3):
            return self.multiply(other=other)
        else:
            raise ValueError(f"Unsupported argument type for @ operator: {type(other)}")

    def jlog(self):
        """Derivatve of log(this.inv() * x) by x at x=this."""
        theta = self.log()

        theta_squared = np.sum(np.square(theta), axis=-1)
        use_taylor = theta_squared < get_epsilon(theta_squared.dtype)

        # Shim to avoid NaNs in np.where branches, which cause failures for
        # reverse-mode AD.
        theta_squared_safe = np.where(
            use_taylor,
            np.ones_like(theta_squared),  # Any non-zero value should do here.
            theta_squared,
        )
        del theta_squared
        theta_safe = np.sqrt(theta_squared_safe)
        half_theta_safe = theta_safe / 2.0

        skew_omega = skew(theta)
        V_inv = np.where(
            use_taylor[None, None],
            np.eye(3)
            - 0.5 * skew_omega
            + np.einsum("ij,jk->ik", skew_omega, skew_omega) / 12.0,
            (
                np.eye(3)
                - 0.5 * skew_omega
                + (
                    (
                        1.0
                        - theta_safe
                        * np.cos(half_theta_safe)
                        / (2.0 * np.sin(half_theta_safe))
                    )
                    / theta_squared_safe
                )[None, None]
                * np.einsum("ij,jk->ik", skew_omega, skew_omega)
            ),
        )

        return V_inv.T
