from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .so3 import SO3
from .utils import get_epsilon, skew

_IDENTITY_WXYZ_XYZ = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


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
    def from_rotation(rotation: SO3) -> SE3:
        return SE3.from_rotation_and_translation(
            rotation=rotation,
            translation=np.zeros(
                SE3.space_dim,
            ),
        )

    @staticmethod
    def from_translation(translation: np.ndarray) -> SE3:
        return SE3.from_rotation_and_translation(
            rotation=SO3.identity(), translation=translation
        )

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
            translation=np.random.uniform(-1.0, 1.0, size=(SE3.space_dim,)),
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
        rotation = SO3.exp(tangent[3:])
        theta_squared = tangent[3:] @ tangent[3:]
        use_taylor = theta_squared < get_epsilon(theta_squared.dtype)
        theta_squared_safe = 1.0 if use_taylor else theta_squared
        theta_safe = np.sqrt(theta_squared_safe)
        skew_omega = skew(tangent[3:])
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
        use_taylor = theta_squared < get_epsilon(theta_squared.dtype)
        skew_omega = skew(omega)
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
                np.eye(3)
                - 0.5 * skew_omega
                + (
                    1
                    - theta_safe
                    * np.cos(half_theta_safe)
                    / (2 * np.sin(half_theta_safe))
                )
                / (theta_safe * theta_safe)
                * (skew_omega @ skew_omega)
            )
            V_inv_2 = (
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
            np.testing.assert_array_almost_equal(V_inv, V_inv_2)
            # from ipdb import set_trace; set_trace()
        return np.concatenate([V_inv @ self.translation(), omega])

    def adjoint(self) -> np.ndarray:
        R = self.rotation().as_matrix()
        return np.block(
            [
                [R, skew(self.translation()) @ R],
                [np.zeros((3, 3), dtype=np.float64), R],
            ]
        )

    def inverse(self) -> SE3:
        R_inv = self.rotation().inverse()
        return SE3.from_rotation_and_translation(
            rotation=R_inv,
            translation=-(R_inv @ self.translation()),
        )

    def normalize(self) -> SE3:
        return SE3.from_rotation_and_translation(
            rotation=self.rotation().normalize(),
            translation=self.translation(),
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

    def copy(self) -> SE3:
        return SE3(wxyz_xyz=self.wxyz_xyz.copy())

    def jlog(self):
        """Derivatve of log(this.inv() * x) by x at x=this."""
        rotation = self.rotation()
        translation = self.translation()
        jlog_so3 = rotation.jlog()
        w = rotation.log()
        theta = np.linalg.norm(w)
        use_taylor = theta < get_epsilon(theta.dtype)
        t2 = theta * theta
        tinv = 1 / theta
        t2inv = tinv * tinv
        st, ct = np.sin(theta), np.cos(theta)
        inv_2_2ct = 1 / (2 * (1 - ct))
        beta = np.where(use_taylor, 1 / 12 + t2 / 720, t2inv - st * tinv * inv_2_2ct)

        beta_dot_over_theta = np.where(
            use_taylor,
            1 / 360,
            -2 * t2inv * t2inv + (1 + st * tinv) * t2inv * inv_2_2ct,
        )
        wTp = w @ translation
        v3_tmp = (beta_dot_over_theta * wTp) * w - (
            theta**2 * beta_dot_over_theta + 2 * beta
        ) * translation
        C = (
            np.outer(v3_tmp, w)
            + beta * np.outer(w, translation)
            + wTp * beta * np.eye(3)
        )
        C = C + 0.5 * skew(translation)
        B = C @ jlog_so3
        jlog = np.zeros((6, 6))
        jlog[:3, :3] = jlog_so3
        jlog[3:, 3:] = jlog_so3
        jlog[:3, 3:] = B
        return jlog
