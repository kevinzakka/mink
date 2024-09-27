"""Tests for configuration_limit.py."""

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.limits import ConfigurationLimit, VelocityLimit
from mink.limits.exceptions import LimitDefinitionError
from mink.utils import get_freejoint_dims


class TestConfigurationLimit(absltest.TestCase):
    """Test configuration limit."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("g1_mj_description")

    def setUp(self):
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("stand")
        # NOTE(kevin): These velocities are arbitrary and do not match real hardware.
        self.velocities = {
            self.model.joint(i).name: 3.14 for i in range(1, self.model.njnt)
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
        nb = nv - len(get_freejoint_dims(self.model)[1])
        self.assertEqual(len(limit.indices), nb)
        self.assertEqual(limit.projection_matrix.shape, (nb, nv))

    def test_indices(self):
        limit = ConfigurationLimit(self.model)
        expected = np.arange(6, self.model.nv)  # Freejoint (0-5) is not limited.
        self.assertTrue(np.allclose(limit.indices, expected))

    def test_model_with_no_limit(self):
        empty_model = mujoco.MjModel.from_xml_string("<mujoco></mujoco>")
        empty_bounded = ConfigurationLimit(empty_model)
        self.assertEqual(len(empty_bounded.indices), 0)
        self.assertIsNone(empty_bounded.projection_matrix)
        G, h = empty_bounded.compute_qp_inequalities(self.configuration, 1e-3)
        self.assertIsNone(G)
        self.assertIsNone(h)

    def test_model_with_subset_of_velocities_limited(self):
        xml_str = """
        <mujoco>
          <compiler angle="radian"/>
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
        nb = 1  # 1 limited joint.
        nv = model.nv
        self.assertEqual(limit.projection_matrix.shape, (nb, nv))
        self.assertEqual(len(limit.indices), nb)
        expected_lower = np.array([-mujoco.mjMAXVAL, 0])
        expected_upper = np.array([mujoco.mjMAXVAL, 1.57])
        np.testing.assert_allclose(limit.lower, expected_lower)
        np.testing.assert_allclose(limit.upper, expected_upper)

    def test_freejoint_ignored(self):
        xml_str = """
        <mujoco>
          <compiler angle="radian"/>
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
        nb = 1  # 1 limited joint.
        nv = model.nv
        self.assertEqual(limit.projection_matrix.shape, (nb, nv))
        self.assertEqual(len(limit.indices), nb)
        expected_lower = np.asarray(
            [
                -mujoco.mjMAXVAL,
            ]
            * 7
            + [0]
        )
        expected_upper = np.asarray(
            [
                mujoco.mjMAXVAL,
            ]
            * 7
            + [1.57]
        )
        np.testing.assert_allclose(limit.lower, expected_lower)
        np.testing.assert_allclose(limit.upper, expected_upper)

    def test_far_from_limit(self, tol=1e-10):
        """Limit has no effect when the configuration is far away."""
        dt = 1e-3  # [s]
        model = load_robot_description("ur5e_mj_description")
        configuration = Configuration(model)
        limit = ConfigurationLimit(model)
        G, h = limit.compute_qp_inequalities(configuration, dt=dt)
        velocities = {
            "shoulder_pan_joint": np.pi,
            "shoulder_lift_joint": np.pi,
            "elbow_joint": np.pi,
            "wrist_1_joint": np.pi,
            "wrist_2_joint": np.pi,
            "wrist_3_joint": np.pi,
        }
        vel_limit = VelocityLimit(model, velocities)
        self.assertLess(np.max(+G @ vel_limit.limit * dt - h), -tol)
        self.assertLess(np.max(-G @ vel_limit.limit * dt - h), -tol)

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
