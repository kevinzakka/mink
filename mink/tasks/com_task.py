import mujoco
import numpy as np

from mink.configuration import Configuration
from mink.tasks import Task


class ComTask(Task):
    def __init__(
        self,
        cost: np.ndarray,
        gain: float = 1.0,
        lm_damping: float = 0.0,
        target_com: np.ndarray | None = None,
    ):
        super().__init__(cost=np.full((3,), cost), gain=gain, lm_damping=lm_damping)

        self.target_com = target_com

    def set_target(self, target_com: np.ndarray) -> None:
        self.target_com = target_com.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        desired_com = configuration.data.subtree_com[1]
        self.set_target(desired_com)

    def set_target_from_mocap(self, data: mujoco.MjData, mocap_id: int) -> None:
        self.set_target(data.mocap_pos[mocap_id])

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        if self.target_com is None:
            raise ValueError("Target COM is not set.")

        error = configuration.data.subtree_com[1] - self.target_com
        return error

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        J = np.empty((3, configuration.model.nv))
        mujoco.mj_jacSubtreeCom(configuration.model, configuration.data, J, 1)
        return J
