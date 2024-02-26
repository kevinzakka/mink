from __future__ import annotations

import numpy as np
from dataclasses import dataclass
import mujoco

_IDENTITIY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
_IDENTITY_WXYZ_XYZ = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
_INVERT_QUAT_SIGN = np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float64)


def _skew(x: np.ndarray) -> np.ndarray:
    assert x.shape == (3,)
    wx, wy, wz = x
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ]
    )


def _get_epsilon(dtype: np.dtype) -> float:
    return {
        np.dtype("float32"): 1e-5,
        np.dtype("float64"): 1e-10,
    }[dtype]


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

    def __repr__(self) -> str:
        wxyz = np.round(self.wxyz, 5)
        return f"{self.__class__.__name__}(wxyz={wxyz})"

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
            low=np.zeros(shape=(3,), dtype=np.float64),
            high=np.array([1.0, 2.0 * np.pi, 2.0 * np.pi]),
            shape=(3,),
        )
        a = np.sqrt(1.0 - u1)
        b = np.sqrt(u1)
        wxyz = np.array(
            [
                a * np.sin(u2),
                a * np.cos(u2),
                b * np.sin(u3),
                b * np.cos(u3),
            ]
        )
        return SO3(wxyz=wxyz)

    def as_matrix(self) -> np.ndarray:
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, self.wxyz)
        return mat.reshape(3, 3)

    @staticmethod
    def exp(tangent: np.ndarray) -> SO3:
        assert tangent.shape == (SO3.tangent_dim,)
        theta_squared = tangent @ tangent
        theta_pow_4 = theta_squared * theta_squared
        use_taylor = theta_squared < _get_epsilon(tangent.dtype)
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
        use_taylor = norm_sq < _get_epsilon(norm_sq.dtype)
        norm_safe = 1.0 if use_taylor else np.sqrt(norm_sq)
        w_safe = w if use_taylor else 1.0
        atan_n_over_w = np.arctan2(-norm_safe if w < 0 else norm_safe, abs(w))
        if use_taylor:
            atan_factor = 2.0 / w_safe - 2.0 / 3.0 * norm_sq / w_safe**3
        else:
            if abs(w) < _get_epsilon(w.dtype):
                scl = 1.0 if w > 0.0 else -1.0
                atan_factor = scl * np.pi / norm_safe
            else:
                atan_factor = 2.0 * atan_n_over_w / norm_safe
        return atan_factor * self.wxyz[1:]

    def adjoint(self) -> np.ndarray:
        return self.as_matrix()

    def inverse(self) -> SO3:
        return SO3(wxyz=self.wxyz * _INVERT_QUAT_SIGN)

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
                ]
            )
        )

    def __matmul__(self, other: SO3 | np.ndarray) -> SO3 | np.ndarray:
        if isinstance(other, np.ndarray):
            return self.apply(target=other)
        elif isinstance(other, SO3):
            return self.multiply(other=other)
        else:
            raise ValueError(f"Unsupported argument type for @ operator: {type(other)}")


@dataclass(frozen=True)
class SE3:
    """Special Euclidean group for proper rigid transforms in 3D.

    Internal parameterization is (qw, qx, qy, qz, x, y, z). Tangent parameterization is
    (vx, vy, vz, omega_x, omega_y, omega_z).
    """

    wxyz_xyz: np.ndarray
    matrix_dim: int = 4
    parameters_dim: int = 7
    tangent_dim: int = 6
    space_dim: int = 3

    def __repr__(self) -> str:
        quat = np.round(self.wxyz_xyz[:4], 5)
        xyz = np.round(self.wxyz_xyz[4:], 5)
        return f"{self.__class__.__name__}(wxyz={quat}, xyz={xyz})"

    @staticmethod
    def identity() -> SE3:
        return SE3(wxyz_xyz=_IDENTITY_WXYZ_XYZ)

    @staticmethod
    def from_rotation_and_translation(
        rotation: SO3,
        translation: np.ndarray,
    ) -> SE3:
        assert translation.shape == (SE3.space_dim,)
        return SE3(wxyz_xyz=np.concatenate([rotation.wxyz, translation]))

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> SE3:
        assert matrix.shape == (SE3.matrix_dim, SE3.matrix_dim)
        return SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(matrix[:3, :3]),
            translation=matrix[:3, 3],
        )

    @staticmethod
    def sample_uniform() -> SE3:
        return SE3.from_rotation_and_translation(
            rotation=SO3.sample_uniform(),
            translation=np.random.uniform(
                -1.0, 1.0, size=(SE3.space_dim,), dtype=np.float64
            ),
        )

    def rotation(self) -> SO3:
        return SO3(wxyz=self.wxyz_xyz[:4])

    def translation(self) -> np.ndarray:
        return self.wxyz_xyz[4:]

    def as_matrix(self) -> np.ndarray:
        hmat = np.eye(self.matrix_dim, dtype=np.float64)
        hmat[:3, :3] = self.rotation().as_matrix()
        hmat[:3, 3] = self.translation()
        return hmat

    @staticmethod
    def exp(tangent: np.ndarray) -> SE3:
        assert tangent.shape == (SE3.tangent_dim,)
        rotation = SO3.exp(tangent[:3])
        theta_squared = tangent[3:] @ tangent[3:]
        use_taylor = theta_squared < _get_epsilon(theta_squared.dtype)
        theta_squared_safe = 1.0 if use_taylor else theta_squared
        theta_safe = np.sqrt(theta_squared_safe)
        skew_omega = _skew(tangent[3:])
        if use_taylor:
            V = rotation.as_matrix()
        else:
            V = (
                np.eye(3, dtype=np.float64)
                + (1.0 - np.cos(theta_safe)) / (theta_squared_safe) * skew_omega
                + (theta_safe - np.sin(theta_safe))
                / (theta_squared_safe * theta_safe)
                * (skew_omega @ skew_omega)
            )
        return SE3.from_rotation_and_translation(
            rotation=rotation,
            translation=V @ tangent[:3],
        )

    def log(self) -> np.ndarray:
        omega = self.rotation().log()
        theta_squared = omega @ omega
        use_taylor = theta_squared < _get_epsilon(theta_squared.dtype)
        skew_omega = _skew(omega)
        theta_squared_safe = 1.0 if use_taylor else theta_squared
        theta_safe = np.sqrt(theta_squared_safe)
        half_theta_safe = 0.5 * theta_safe
        if use_taylor:
            V_inv = (
                np.eye(3, dtype=np.float64)
                - 0.5 * skew_omega
                + (skew_omega @ skew_omega) / 12.0
            )
        else:
            V_inv = (
                np.eye(3, dtype=np.float64)
                - 0.5 * skew_omega
                + (
                    1.0
                    - theta_safe
                    * np.cos(half_theta_safe)
                    / (2.0 * np.sin(half_theta_safe))
                )
                / theta_squared_safe
                * (skew_omega @ skew_omega)
            )
        return np.concatenate([V_inv @ self.translation(), omega])

    def adjoint(self) -> np.ndarray:
        R = self.rotation().as_matrix()
        return np.block(
            [
                [R, np.zeros((3, 3), dtype=np.float64)],
                [_skew(self.translation()) @ R, R],
            ]
        )

    def inverse(self) -> SE3:
        R_inv = self.rotation().inverse()
        return SE3.from_rotation_and_translation(
            rotation=R_inv,
            translation=-(R_inv @ self.translation()),
        )

    def apply(self, target: np.ndarray) -> np.ndarray:
        assert target.shape == (SE3.space_dim,)
        return self.rotation() @ target + self.translation()

    def multiply(self, other: SE3) -> SE3:
        return SE3.from_rotation_and_translation(
            rotation=self.rotation() @ other.rotation(),
            translation=(self.rotation() @ other.translation()) + self.translation(),
        )

    def __matmul__(self, other: SE3 | np.ndarray) -> SE3 | np.ndarray:
        if isinstance(other, np.ndarray):
            return self.apply(target=other)
        elif isinstance(other, SE3):
            return self.multiply(other=other)
        else:
            raise ValueError(f"Unsupported argument type for @ operator: {type(other)}")
