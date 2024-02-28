from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import mujoco

from mink.limits import Limit, BoxConstraint


@dataclass(frozen=True)
class VelocityLimit(Limit):
    limit: np.ndarray

    @staticmethod
    def initialize(
        model: mujoco.MjModel,
        joint2limit: dict[str, float],
    ) -> VelocityLimit:
        for joint in joint2limit.keys():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if joint_id == -1:
                raise ValueError(f"Joint '{joint}' not found in the model.")
        return VelocityLimit(
            limit=np.asarray(list(joint2limit.values())),
        )

    def compute_qp_inequalities(self, q: np.ndarray, dt: float) -> BoxConstraint:
        del q  # Unused.
        lower = -dt * self.limit
        upper = dt * self.limit
        return BoxConstraint(lower=lower, upper=upper)
