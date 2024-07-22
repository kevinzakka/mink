"""Tests for com_task.py."""

import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.tasks import (
    ComTask,
    InvalidTarget,
    TargetNotSet,
    TaskDefinitionError,
)


class TestComTask(absltest.TestCase):
    """Test consistency of the com task."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("g1_mj_description")

    def setUp(self):
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("stand")

    def test_cost_correctly_broadcast(self):
        task = ComTask(cost=1.0)
        np.testing.assert_array_equal(task.cost, np.ones((3,)))
        task = ComTask(cost=[5.0])
        np.testing.assert_array_equal(task.cost, np.full((3,), 5.0))
        task = ComTask(cost=[1, 2, 3])
        np.testing.assert_array_equal(task.cost, np.asarray([1, 2, 3]))

    def test_task_raises_error_if_cost_dim_invalid(self):
        with self.assertRaises(TaskDefinitionError) as cm:
            ComTask(cost=(1, 2))
        expected_error_message = (
            "ComTask cost must be a vector of shape (1,) (aka identical cost for all "
            "coordinates) or (3,). Got (2,)"
        )
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_task_raises_error_if_cost_negative(self):
        with self.assertRaises(TaskDefinitionError) as cm:
            ComTask(cost=(-1))
        self.assertEqual(str(cm.exception), "ComTask cost must be >= 0")

    def test_task_raises_error_if_target_is_invalid(self):
        task = ComTask(cost=1.0)
        with self.assertRaises(InvalidTarget) as cm:
            task.set_target(np.random.rand(5))
        expected_error_message = "Expected target CoM to have shape (3,) but got (5,)"
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_error_without_target(self):
        task = ComTask(cost=1.0)
        with self.assertRaises(TargetNotSet) as cm:
            task.compute_error(self.configuration)
        self.assertEqual(str(cm.exception), "No target set for ComTask")

    def test_jacobian_without_target(self):
        task = ComTask(cost=1.0)
        with self.assertRaises(TargetNotSet):
            task.compute_jacobian(self.configuration)

    def test_set_target_from_configuration(self):
        task = ComTask(cost=1.0)
        task.set_target_from_configuration(self.configuration)
        com = self.configuration.data.subtree_com[1]
        np.testing.assert_array_equal(task.target_com, com)

    def test_target_is_a_copy(self):
        task = ComTask(cost=1.0)
        com_desired = np.zeros(3)
        task.set_target(com_desired)
        com_desired[0] = 1.0
        np.testing.assert_array_equal(task.target_com, np.zeros(3))

    def test_zero_error_when_target_at_body(self):
        task = ComTask(cost=1.0)
        task.set_target_from_configuration(self.configuration)
        error = task.compute_error(self.configuration)
        np.testing.assert_allclose(error, np.zeros(3))

    def test_zero_cost_same_as_disabling_task(self):
        task = ComTask(cost=0.0)
        task.set_target_from_configuration(self.configuration)
        objective = task.compute_qp_objective(self.configuration)
        x = np.random.random(self.configuration.nv)
        cost = objective.value(x)
        self.assertAlmostEqual(cost, 0.0)


if __name__ == "__main__":
    absltest.main()
