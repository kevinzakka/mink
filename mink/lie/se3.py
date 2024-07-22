from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..exceptions import InvalidMocapBody
from .base import MatrixLieGroup
from .so3 import SO3
from .utils import get_epsilon, skew

_IDENTITY_WXYZ_XYZ = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class SE3(MatrixLieGroup):
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

    def copy(self) -> SE3:
        return SE3(wxyz_xyz=np.array(self.wxyz_xyz))

    def parameters(self) -> np.ndarray:
        return self.wxyz_xyz

    @classmethod
    def identity(cls) -> SE3:
        return SE3(wxyz_xyz=_IDENTITY_WXYZ_XYZ)

    @classmethod
    def from_rotation_and_translation(
        cls,
        rotation: SO3,
        translation: np.ndarray,
    ) -> SE3:
        assert translation.shape == (SE3.space_dim,)
        return SE3(wxyz_xyz=np.concatenate([rotation.wxyz, translation]))

    @classmethod
    def from_rotation(cls, rotation: SO3) -> SE3:
        return SE3.from_rotation_and_translation(
            rotation=rotation,
            translation=np.zeros(
                SE3.space_dim,
            ),
        )

    @classmethod
    def from_translation(cls, translation: np.ndarray) -> SE3:
        return SE3.from_rotation_and_translation(
            rotation=SO3.identity(), translation=translation
        )

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> SE3:
        assert matrix.shape == (SE3.matrix_dim, SE3.matrix_dim)
        return SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(matrix[:3, :3]),
            translation=matrix[:3, 3],
        )

    @classmethod
    def from_mocap_id(cls, data: mujoco.MjData, mocap_id: int) -> SE3:
        return SE3.from_rotation_and_translation(
            rotation=SO3(data.mocap_quat[mocap_id]),
            translation=data.mocap_pos[mocap_id],
        )

    @classmethod
    def from_mocap_name(
        cls, model: mujoco.MjModel, data: mujoco.MjData, mocap_name: str
    ) -> SE3:
        mocap_id = model.body(mocap_name).mocapid[0]
        if mocap_id == -1:
            raise InvalidMocapBody(mocap_name, model)
        return SE3.from_mocap_id(data, mocap_id)

    @classmethod
    def sample_uniform(cls) -> SE3:
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

    @classmethod
    def exp(cls, tangent: np.ndarray) -> SE3:
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

    def log(self) -> np.ndarray:
        omega = self.rotation().log()
        theta_squared = omega @ omega
        use_taylor = theta_squared < get_epsilon(theta_squared.dtype)
        skew_omega = skew(omega)
        theta_squared_safe = 1.0 if use_taylor else theta_squared
        theta_safe = np.sqrt(theta_squared_safe)
        half_theta_safe = 0.5 * theta_safe
        skew_omega_norm = skew_omega @ skew_omega
        if use_taylor:
            V_inv = (
                np.eye(3, dtype=np.float64) - 0.5 * skew_omega + skew_omega_norm / 12.0
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
                * skew_omega_norm
            )
        return np.concatenate([V_inv @ self.translation(), omega])

    def adjoint(self) -> np.ndarray:
        R = self.rotation().as_matrix()
        return np.block(
            [
                [R, skew(self.translation()) @ R],
                [np.zeros((3, 3), dtype=np.float64), R],
            ]
        )

    # Jacobians.

    # Eqn 179 a)
    @classmethod
    def ljac(cls, other: np.ndarray) -> np.ndarray:
        theta = other[3:]
        if theta @ theta < get_epsilon(theta.dtype):
            return np.eye(cls.tangent_dim)
        Q = _getQ(other)
        J = SO3.ljac(theta)
        O = np.zeros((3, 3))
        return np.block([[J, Q], [O, J]])

    # Eqn 179 b)
    @classmethod
    def ljacinv(cls, other: np.ndarray) -> np.ndarray:
        theta = other[3:]
        if theta @ theta < get_epsilon(theta.dtype):
            return np.eye(cls.tangent_dim)
        Q = _getQ(other)
        J_inv = SO3.ljacinv(theta)
        O = np.zeros((3, 3))
        return np.block([[J_inv, -J_inv @ Q @ J_inv], [O, J_inv]])


# Eqn 180.
def _getQ(c) -> np.ndarray:
    theta_sq = c[3:] @ c[3:]
    A = 0.5
    if theta_sq < get_epsilon(theta_sq.dtype):
        B = (1.0 / 6.0) + (1.0 / 120.0) * theta_sq
        C = -(1.0 / 24.0) + (1.0 / 720.0) * theta_sq
        D = -(1.0 / 60.0)
    else:
        theta = np.sqrt(theta_sq)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        B = (theta - sin_theta) / (theta_sq * theta)
        C = (1.0 - theta_sq / 2.0 - cos_theta) / (theta_sq * theta_sq)
        D = ((2) * theta - (3) * sin_theta + theta * cos_theta) / (
            (2) * theta_sq * theta_sq * theta
        )
    V = skew(c[:3])
    W = skew(c[3:])
    VW = V @ W
    WV = VW.T
    WVW = WV @ W
    VWW = VW @ W
    return (
        +A * V
        + B * (WV + VW + WVW)
        - C * (VWW - VWW.T - 3 * WVW)
        + D * (WVW @ W + W @ WVW)
    )
