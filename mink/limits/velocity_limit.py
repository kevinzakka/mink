from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import mujoco

from mink.limits import Limit, Inequality


@dataclass(frozen=True)
class VelocityLimit(Limit):
    indices: np.ndarray
    projection_matrix: np.ndarray
    limit: np.ndarray

    @staticmethod
    def initialize(
        model: mujoco.MjModel,
        joint2limit: dict[str, float],
    ) -> VelocityLimit:
        indices = []
        for joint in joint2limit.keys():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if joint_id == -1:
                raise ValueError(f"Joint '{joint}' not found in the model.")
            indices.append(joint_id)

        return VelocityLimit(
            indices=np.asarray(indices),
            projection_matrix=np.eye(model.nv)[indices],
            limit=np.asarray(list(joint2limit.values())),
        )

    def compute_qp_inequalities(self, q: np.ndarray, dt: float) -> Inequality:
        del q  # Unused.
        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        h = np.hstack([dt * self.limit, dt * self.limit])
        return Inequality(G, h)
