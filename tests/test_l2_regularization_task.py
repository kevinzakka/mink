"""Tests for l2_regularization_task.py."""

import numpy as np
from absl.testing import absltest

from mink.exceptions import TaskDefinitionError
from mink.tasks import L2RegularizationTask
from mink import Configuration
from robot_descriptions.loaders.mujoco import load_robot_description


class TestL2RegularizationTask(absltest.TestCase):
    """Test consistency of the L2 regularization task."""

    def test_cost_must_be_nonnegative(self):
        with self.assertRaises(TaskDefinitionError) as cm:
            L2RegularizationTask(cost=-1.0)
        self.assertEqual(str(cm.exception), "L2RegularizationTask cost should be >= 0")

    def test_cost_must_be_scalar(self):
        with self.assertRaises(TaskDefinitionError) as cm:
            L2RegularizationTask(cost=[1.0, 2.0])
        self.assertEqual(
            str(cm.exception), "L2RegularizationTask cost must be a scalar"
        )

    def test_qp_objective_is_correct(self):
        model = load_robot_description("g1_mj_description")
        configuration = Configuration(model)
        task = L2RegularizationTask(cost=1e-3)
        objective = task.compute_qp_objective(configuration)
        # All dofs are regularized equally, including floating-base coordinates.
        np.testing.assert_array_equal(objective.H, np.eye(configuration.nv) * 1e-3)
        np.testing.assert_array_equal(objective.c, np.zeros(configuration.nv))


if __name__ == "__main__":
    absltest.main()
