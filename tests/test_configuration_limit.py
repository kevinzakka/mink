"""Tests for configuration_limit.py."""

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.limits import ConfigurationLimit, VelocityLimit
from mink.limits.exceptions import LimitDefinitionError


class TestConfigurationLimit(absltest.TestCase):
    """Test configuration limit."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self):
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("home")
        self.velocities = {
            "shoulder_pan_joint": np.pi,
            "shoulder_lift_joint": np.pi,
            "elbow_joint": np.pi,
            "wrist_1_joint": np.pi,
            "wrist_2_joint": np.pi,
            "wrist_3_joint": np.pi,
        }
        self.vel_limit = VelocityLimit(self.model, self.velocities)

    def test_throws_error_if_gain_invalid(self):
        with self.assertRaises(LimitDefinitionError):
            ConfigurationLimit(self.model, gain=-1)
        with self.assertRaises(LimitDefinitionError):
            ConfigurationLimit(self.model, gain=1.1)

    def test_dimensions(self):
        limit = ConfigurationLimit(self.model)
        nv = self.configuration.nv
        self.assertEqual(limit.projection_matrix.shape, (nv, nv))
        self.assertEqual(len(limit.indices), nv)

    def test_model_with_no_limit(self):
        empty_model = mujoco.MjModel.from_xml_string("<mujoco></mujoco>")
        empty_bounded = ConfigurationLimit(empty_model)
        self.assertEqual(len(empty_bounded.indices), 0)
        self.assertIsNone(empty_bounded.projection_matrix)

    def test_model_with_subset_of_velocities_limited(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body>
              <joint type="hinge" name="hinge_unlimited"/>
              <geom type="sphere" size=".1" mass=".1"/>
              <body>
                <joint type="hinge" name="hinge_limited" limited="true" range="0 1.57"/>
                <geom type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        limit = ConfigurationLimit(model)
        nb = 1
        nv = model.nv
        self.assertEqual(limit.projection_matrix.shape, (nb, nv))
        self.assertEqual(len(limit.indices), nb)

    def test_freejoint_ignored(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body>
              <joint type="free" name="floating"/>
              <geom type="sphere" size=".1" mass=".1"/>
              <body>
                <joint type="hinge" name="hinge" range="0 1.57" limited="true"/>
                <geom type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        limit = ConfigurationLimit(model)
        nb = 1
        nv = model.nv
        self.assertEqual(limit.projection_matrix.shape, (nb, nv))
        self.assertEqual(len(limit.indices), nb)

    def test_far_from_limit(self, tol=1e-10):
        """Limit has no effect when the configuration is far away."""
        dt = 1e-3  # [s]
        limit = ConfigurationLimit(self.model)
        G, h = limit.compute_qp_inequalities(self.configuration, dt=dt)
        # When we are far away from configuration limits, the velocity limit is
        # simply the configuration-agnostic one from the robot.
        self.assertLess(np.max(+G @ self.vel_limit.limit * dt - h), -tol)
        self.assertLess(np.max(-G @ self.vel_limit.limit * dt - h), -tol)

    def test_configuration_limit_repulsion(self, tol=1e-10):
        """Velocities are scaled down when close to a configuration limit."""
        dt = 1e-3  # [s]
        slack_vel = 5e-4  # [rad] / [s]
        limit = ConfigurationLimit(self.model, gain=0.5)
        # Override configuration limits to `q +/- slack_vel * dt`.
        limit.lower = self.configuration.integrate(
            -slack_vel * np.ones((self.configuration.nv,)), dt
        )
        limit.upper = self.configuration.integrate(
            +slack_vel * np.ones((self.configuration.nv,)), dt
        )
        _, h = limit.compute_qp_inequalities(self.configuration, dt)
        self.assertLess(np.max(h), slack_vel * dt + tol)
        self.assertGreater(np.min(h), -slack_vel * dt - tol)


if __name__ == "__main__":
    absltest.main()
