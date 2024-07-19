"""Tests for utils.py."""

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import utils


class TestUtils(absltest.TestCase):
    """Test utility functions."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("stretch_mj_description")

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        self.q0 = self.data.qpos.copy()

    def test_custom_configuration_vector(self):
        custom_joints = dict(
            joint_head_pan=0.2,  # Hinge.
            joint_arm_l3=0.1,  # Slide.
        )
        q = utils.custom_configuration_vector(self.model, **custom_joints)
        q_expected = self.q0.copy()
        for name, value in custom_joints.items():
            qid = self.model.jnt_dofadr[self.model.joint(name).id]
            q_expected[qid] = value
        np.testing.assert_array_almost_equal(q, q_expected)


if __name__ == "__main__":
    absltest.main()
