"""Tests for posture_task.py."""

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.exceptions import InvalidTarget, TargetNotSet, TaskDefinitionError
from mink.tasks import PostureTask


class TestPostureTask(absltest.TestCase):
    """Test consistency of the posture task."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("talos_mj_description")

    def setUp(self):
        self.configuration = Configuration(self.model)

    def test_task_raises_error_if_cost_negative(self):
        with self.assertRaises(TaskDefinitionError) as cm:
            PostureTask(model=self.model, cost=(-1))
        self.assertEqual(str(cm.exception), "PostureTask cost should be >= 0")

    def test_cost_correctly_broadcast(self):
        task = PostureTask(model=self.model, cost=5.0)
        np.testing.assert_array_equal(task.cost, np.ones((self.model.nv,)) * 5.0)
        task = PostureTask(model=self.model, cost=[5.0])
        np.testing.assert_array_equal(task.cost, np.ones((self.model.nv,)) * 5.0)
        cost = np.random.random(size=(self.model.nv,))
        task = PostureTask(model=self.model, cost=cost)
        np.testing.assert_array_equal(task.cost, cost)

    def test_cost_invalid_shape(self):
        with self.assertRaises(TaskDefinitionError) as cm:
            PostureTask(model=self.model, cost=(0.5, 2.0))
        expected_error_message = (
            "PostureTask cost must be a vector of shape (1,) (aka identical cost for "
            f"all dofs) or ({self.model.nv},). Got (2,)"
        )
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_task_raises_error_if_target_is_invalid(self):
        task = PostureTask(model=self.model, cost=1.0)
        with self.assertRaises(InvalidTarget) as cm:
            task.set_target(np.random.rand(1))
        expected_error_message = (
            f"Expected target posture to have shape ({self.model.nq},) but got (1,)"
        )
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_error_without_target(self):
        task = PostureTask(model=self.model, cost=1.0)
        with self.assertRaises(TargetNotSet) as cm:
            task.compute_error(self.configuration)
        self.assertEqual(str(cm.exception), "No target set for PostureTask")

    def test_ball_joint_jacobian_matches_finite_differences(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><joint type="ball"/>'
            '<geom type="sphere" size=".1"/></body></worldbody></mujoco>'
        )
        target = np.zeros(4)
        mujoco.mju_axisAngle2Quat(target, np.array([1.0, 0.0, 0.0]), 0.7)
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, np.array([0.0, 1.0, 0.0]), 1.1)
        configuration = Configuration(model, q)
        task = PostureTask(model, cost=1.0)
        task.set_target(target)
        error = task.compute_error(configuration)
        eps = 1e-6
        expected = np.empty((3, 3))
        for i in range(3):
            dq = np.zeros(3)
            dq[i] = eps
            q_pert = q.copy()
            mujoco.mj_integratePos(model, q_pert, dq, 1.0)
            error_pert = task.compute_error(Configuration(model, q_pert))
            expected[:, i] = (error_pert - error) / eps
        np.testing.assert_allclose(
            task.compute_jacobian(configuration), expected, atol=1e-5
        )

    def test_jacobian_can_be_computed_without_target(self):
        task = PostureTask(model=self.model, cost=1.0)
        jac = task.compute_jacobian(self.configuration)
        eye = np.eye(self.configuration.nv)
        eye[:6, :6] = 0.0
        np.testing.assert_allclose(jac, eye)

    def test_set_target_from_configuration(self):
        task = PostureTask(model=self.model, cost=1.0)
        task.set_target_from_configuration(self.configuration)
        np.testing.assert_array_equal(task.target_q, self.configuration.q)

    def test_target_is_a_copy(self):
        task = PostureTask(model=self.model, cost=1.0)
        q_desired = np.zeros((self.model.nq,))
        task.set_target(q_desired)
        q_desired[0] = 1.0
        np.testing.assert_array_equal(task.target_q, np.zeros((self.model.nq,)))

    def test_zero_error_when_target_is_current_configuration(self):
        task = PostureTask(model=self.model, cost=1.0)
        task.set_target_from_configuration(self.configuration)
        error = task.compute_error(self.configuration)
        np.testing.assert_allclose(error, np.zeros(self.model.nv))

    def test_unit_cost_qp_objective(self):
        """Unit cost means the QP objective is exactly (J^T J, -e^T J)."""
        task = PostureTask(model=self.model, cost=1.0)
        task.set_target_from_configuration(self.configuration)
        q_new = self.configuration.q.copy()
        q_new[1] += 1.0
        q_new[3] += 1.0
        q_new[5] += 1.0
        self.configuration.update(q=q_new)
        J = task.compute_jacobian(self.configuration)
        e = task.compute_error(self.configuration)
        H, c = task.compute_qp_objective(self.configuration)
        np.testing.assert_allclose(H, J.T @ J)
        np.testing.assert_allclose(c, e.T @ J)

    def test_zero_cost_same_as_disabling_task(self):
        task = PostureTask(model=self.model, cost=0.0)
        task.set_target_from_configuration(self.configuration)
        objective = task.compute_qp_objective(self.configuration)
        x = np.random.random(self.configuration.nv)
        cost = objective.value(x)
        self.assertAlmostEqual(cost, 0.0)


if __name__ == "__main__":
    absltest.main()
