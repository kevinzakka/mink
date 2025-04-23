"""Test task jacobian matrices against finite differences."""

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

import mink
from mink import lie

_TOL = 1e-5
_STEP_SIZE = np.sqrt(np.finfo(float).eps)


class TestJacobians(absltest.TestCase):
    """Test task jacobian matrices against finite differences."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("talos_mj_description")

    def setUp(self, nb_configs: int = 1):
        np.random.seed(42)

        lower = np.full(self.model.nq, -2 * np.pi)
        upper = np.full(self.model.nq, 2 * np.pi)
        for jnt in range(self.model.njnt):
            jnt_type = self.model.jnt_type[jnt]
            if jnt_type not in mink.SUPPORTED_FRAMES or not self.model.jnt_limited[jnt]:
                continue
            padr = self.model.jnt_qposadr[jnt]
            lower[padr : padr + 1] = self.model.jnt_range[jnt, 0]
            upper[padr : padr + 1] = self.model.jnt_range[jnt, 1]

        random_q = np.random.uniform(
            low=lower,
            high=upper,
            size=(nb_configs, self.model.nq),
        )
        self.random_q = random_q

        self.target_q = np.random.uniform(low=lower, high=upper)

    def check_jacobian_finite_diff(self, task: mink.Task, tol: float):
        """Check that a task Jacobian is de/dq by finite differences.

        Args:
            task: Task to test the Jacobian of.
            tol: Test tolerance.
        """
        configuration = mink.Configuration(self.model)

        def e(q) -> np.ndarray:
            configuration.update(q)
            return task.compute_error(configuration)

        def J(q) -> np.ndarray:
            configuration.update(q)
            return task.compute_jacobian(configuration)

        for q_0 in self.random_q:
            J_0 = J(q_0)
            e_0 = e(q_0)
            J_finite = np.empty_like(J_0)
            for i in range(self.model.nv):
                e_i = np.eye(self.model.nv)[i]
                q_perturbed = q_0.copy()
                mujoco.mj_integratePos(self.model, q_perturbed, e_i, _STEP_SIZE)
                J_finite[:, i] = (e(q_perturbed) - e_0) / _STEP_SIZE
            self.assertLess(np.linalg.norm(J_0 - J_finite, ord=np.inf), tol)

    def test_frame_task(self):
        frame_task = mink.FrameTask(
            frame_name="left_foot",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        frame_task.set_target(lie.SE3.sample_uniform())
        self.check_jacobian_finite_diff(frame_task, tol=_TOL)

    def test_relative_frame_task(self):
        relative_frame_task = mink.RelativeFrameTask(
            frame_name="left_foot",
            frame_type="site",
            root_name="torso_1_link",
            root_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        relative_frame_task.set_target(lie.SE3.sample_uniform())
        self.check_jacobian_finite_diff(relative_frame_task, tol=_TOL)

    def test_posture_task(self):
        posture_task = mink.PostureTask(model=self.model, cost=1.0)
        data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, data)
        q0 = data.qpos.copy()
        target_q = q0 + np.random.randn(self.model.nq) * 1e-3
        posture_task.set_target(target_q)
        self.check_jacobian_finite_diff(posture_task, tol=1e-5)

    def test_com_task(self):
        com_task = mink.ComTask(cost=1.0)
        com_task.set_target(np.zeros(3))
        self.check_jacobian_finite_diff(com_task, tol=_TOL)

    def test_equality_constraint_task(self):
        equality_constraint_task = mink.EqualityConstraintTask(self.model, cost=1.0)
        self.check_jacobian_finite_diff(equality_constraint_task, tol=_TOL)


if __name__ == "__main__":
    absltest.main()
