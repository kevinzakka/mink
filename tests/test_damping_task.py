"""Tests for damping_task.py."""

import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.tasks import DampingTask


class TestDampingTask(absltest.TestCase):
    """Test consistency of the damping task."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self):
        self.configuration = Configuration(self.model)

    def test_qp_objective(self):
        task = DampingTask(self.model, cost=1.0)
        nv = self.configuration.nv
        H, c = task.compute_qp_objective(self.configuration)
        np.testing.assert_allclose(H, np.eye(nv))
        np.testing.assert_allclose(c, np.zeros(nv))


if __name__ == "__main__":
    absltest.main()
