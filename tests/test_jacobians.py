from absl.testing import absltest

import mink
from mink import lie
import numpy as np

from robot_descriptions.loaders.mujoco import load_robot_description


class TestJacobians(absltest.TestCase):
    """Test task jacobian matrices against finite differences."""

    def setUp(self, nb_configs: int = 1):
        np.random.seed(42)
        model = load_robot_description("fr3_mj_description")
        random_q = np.random.uniform(
            low=model.jnt_range[:, 0],
            high=model.jnt_range[:, 1],
            size=(nb_configs, model.nq),
        )
        self.model = model
        self.random_q = random_q

    def check_jacobian_finite_diff(self, task: mink.Task, tol: float):
        """Check that a task Jacobian is de/dq by finite differences.

        Args:
            task: Task to test the Jacobian of.
            tol: Test tolerance.
        """

        def e(q) -> np.ndarray:
            configuration = mink.Configuration(self.model, q)
            return task.compute_error(configuration)

        def J(q) -> np.ndarray:
            configuration = mink.Configuration(self.model, q)
            return task.compute_jacobian(configuration)

        nq = self.model.nq
        nv = self.model.nv
        for q_0 in self.random_q:
            J_0 = J(q_0)
            e_0 = e(q_0)
            J_finite = np.empty((e_0.shape[0], nv))
            for i in range(nq):
                h = 0.000001
                e_i = np.eye(nq)[i]
                J_finite[:, i] = (e(q_0 + h * e_i) - e_0) / h
            self.assertLess(np.linalg.norm(J_0 - J_finite, ord=np.inf), tol)

    def test_frame_task(self):
        frame_task = mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        frame_task.set_target(lie.SE3.sample_uniform())
        self.check_jacobian_finite_diff(frame_task, tol=1e-5)

    def test_posture_task(self):
        posture_task = mink.PostureTask(cost=1.0)
        posture_task.set_target(self.model.key("home").qpos)
        self.check_jacobian_finite_diff(posture_task, tol=1e-6)

    def test_com_task(self):
        com_task = mink.ComTask(cost=1.0)
        com_task.set_target(np.zeros(3))
        self.check_jacobian_finite_diff(com_task, tol=1e-6)


if __name__ == "__main__":
    absltest.main()
