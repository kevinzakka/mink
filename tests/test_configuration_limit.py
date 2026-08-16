"""Tests for configuration_limit.py."""

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description
from scipy.optimize import linprog

from mink import Configuration
from mink.exceptions import LimitDefinitionError
from mink.limits import ConfigurationLimit, VelocityLimit
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
        assert limit.projection_matrix is not None
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
        assert limit.projection_matrix is not None
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
        assert limit.projection_matrix is not None
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

    _BALL_JOINT_XML = """
    <mujoco>
      <compiler angle="radian"/>
      <worldbody>
        <body>
          <joint name="ball" type="ball" limited="true" range="0 1.0"/>
          <geom type="sphere" size=".1" mass=".1"/>
        </body>
      </worldbody>
    </mujoco>
    """

    def test_ball_joint_excluded_from_box_constraint(self):
        model = mujoco.MjModel.from_xml_string(self._BALL_JOINT_XML)
        limit = ConfigurationLimit(model)
        self.assertEqual(len(limit.indices), 0)
        self.assertIsNone(limit.projection_matrix)

    def test_ball_joint_min_distance_exceeding_range_throws(self):
        """A margin that eliminates the ball joint's range fails at construction."""
        model = mujoco.MjModel.from_xml_string(self._BALL_JOINT_XML)
        with self.assertRaises(LimitDefinitionError):
            ConfigurationLimit(model, min_distance_from_limits=1.5)

    def test_ball_joint_angle_row(self):
        gain = 0.5
        model = mujoco.MjModel.from_xml_string(self._BALL_JOINT_XML)
        limit = ConfigurationLimit(model, gain=gain)
        configuration = Configuration(model)

        # At the identity orientation the axis is undefined: the row is zero and the
        # constraint is vacuous with slack gain * max_angle.
        G, h = limit.compute_qp_inequalities(configuration, 1e-3)
        assert G is not None and h is not None
        self.assertEqual(G.shape, (1, model.nv))
        self.assertEqual(h.shape, (1,))
        np.testing.assert_allclose(G, 0.0)
        self.assertAlmostEqual(h[0], gain * 1.0)

        # Rotated 0.4 rad about y: row is the rotation axis, slack shrinks.
        angle = 0.4
        q = np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])
        configuration.update(q=q)
        G, h = limit.compute_qp_inequalities(configuration, 1e-3)
        assert G is not None and h is not None
        np.testing.assert_allclose(G[0], [0.0, 1.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(h[0], gain * (1.0 - angle))

    def test_ball_joint_step_respects_angle_limit(self, tol=1e-9):
        """Saturating the constraint lands exactly on the angle limit."""
        model = mujoco.MjModel.from_xml_string(self._BALL_JOINT_XML)
        limit = ConfigurationLimit(model, gain=1.0)
        configuration = Configuration(model)
        angle = 0.4
        q = np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])
        configuration.update(q=q)
        G, h = limit.compute_qp_inequalities(configuration, 1e-3)
        assert G is not None and h is not None
        # Step along the rotation axis with the maximum allowed magnitude.
        q_next = configuration.integrate(G[0] * h[0], dt=1.0)
        angle_next = 2.0 * np.arctan2(np.linalg.norm(q_next[1:]), abs(q_next[0]))
        self.assertAlmostEqual(angle_next, 1.0, delta=tol)

    def test_ball_and_hinge_joints_combined(self):
        xml_str = """
        <mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <body>
              <joint name="ball" type="ball" limited="true" range="0 1.0"/>
              <geom type="sphere" size=".1" mass=".1"/>
              <body>
                <joint name="hinge" type="hinge" limited="true" range="0 1.57"/>
                <geom type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        limit = ConfigurationLimit(model)
        configuration = Configuration(model)
        # Only the hinge is box-constrained; the ball contributes one angle row.
        self.assertEqual(len(limit.indices), 1)
        G, h = limit.compute_qp_inequalities(configuration, 1e-3)
        assert G is not None and h is not None
        self.assertEqual(G.shape, (3, model.nv))
        self.assertEqual(h.shape, (3,))

    def test_far_from_limit(self, tol=1e-10):
        """Limit has no effect when the configuration is far away."""
        dt = 1e-3  # [s]
        model = load_robot_description("ur5e_mj_description")
        configuration = Configuration(model)
        limit = ConfigurationLimit(model)
        G, h = limit.compute_qp_inequalities(configuration, dt=dt)
        assert G is not None and h is not None
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
        assert h is not None
        self.assertLess(np.max(h), slack_vel * dt + tol)
        self.assertGreater(np.min(h), -slack_vel * dt - tol)

    def test_configuration_limit_dt_invariance(self):
        """Inequalities are invariant to dt."""
        limit = ConfigurationLimit(self.model, gain=0.95)
        G1, h1 = limit.compute_qp_inequalities(self.configuration, dt=1e-3)
        G2, h2 = limit.compute_qp_inequalities(self.configuration, dt=0.2)
        assert G1 is not None and h1 is not None
        assert G2 is not None and h2 is not None
        self.assertTrue(np.allclose(G1, G2))
        self.assertTrue(np.allclose(h1, h2))

    def test_feasible_step_respects_position_bounds(self, tol=1e-9):
        """A strictly feasible Δq (for GΔq ≤ h) keeps q_next inside [lower, upper]."""
        limit = ConfigurationLimit(self.model, gain=1.0)

        # dt is irrelevant for configuration limits; pass anything.
        G, h = limit.compute_qp_inequalities(self.configuration, dt=0.1)
        assert G is not None and h is not None

        # We use linprog to construct a strictly feasible delta_q for the inequality
        # set `G Δq ≤ h - eps``. Shrinking the RHS by eps guarantees strict slack if a
        # solution exists. linprog with a zero cost is a convenient feasibility solver.
        eps = 1e-3
        c = np.zeros(self.configuration.nv)
        res = linprog(c, A_ub=G, b_ub=h - eps, bounds=(None, None), method="highs")
        assert res.success, "Could not find a strictly feasible Δq"
        delta_q = res.x

        # Integrate one second with `v = delta_q` (so Δq = v * 1).
        q_next = self.configuration.integrate(delta_q, dt=1.0)

        # Check the next configuration is inside the true bounds for limited joints.
        lower = limit.lower[limit.indices]
        upper = limit.upper[limit.indices]
        qn = q_next[limit.indices]
        assert np.all(qn <= upper + tol)
        assert np.all(qn >= lower - tol)


if __name__ == "__main__":
    absltest.main()
