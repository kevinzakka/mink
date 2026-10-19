"""Tests for free_joint_velocity_limit.py."""

import mujoco
import numpy as np
from absl.testing import absltest, parameterized
from robot_descriptions.loaders.mujoco import load_robot_description

import mink
from mink import Configuration, FrameTask, FreeJointVelocityLimit, VelocityLimit
from mink.exceptions import LimitDefinitionError
from mink.lie import SE3

_TWO_FREE = """
<mujoco>
  <worldbody>
    <body name="a"><joint type="free" name="fa"/><geom type="sphere" size=".1"/></body>
    <body name="b"><joint type="free" name="fb"/><geom type="sphere" size=".1"/></body>
  </worldbody>
</mujoco>
"""

_NO_FREE = """
<mujoco>
  <worldbody>
    <body><joint type="hinge" name="h"/><geom type="sphere" size=".1"/></body>
  </worldbody>
</mujoco>
"""

_ONE_FREE = """
<mujoco>
  <worldbody>
    <body><freejoint/><geom type="sphere" size=".1"/></body>
  </worldbody>
</mujoco>
"""


def _rotated(configuration: Configuration, quat_wxyz) -> None:
    q = configuration.q.copy()
    q[3:7] = quat_wxyz
    configuration.update(q)


def _inequalities(limit, configuration, dt):
    G, h = limit.compute_qp_inequalities(configuration, dt)
    assert G is not None and h is not None
    return G, h


class TestFreeJointVelocityLimitInputOwnership(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.model = mujoco.MjModel.from_xml_string(_ONE_FREE)
        self.configuration = Configuration(self.model)

    @parameterized.product(
        linear=(False, True), use_view=(False, True), readonly=(False, True)
    )
    def test_array_input_is_copied(self, linear, use_view, readonly):
        values = np.arange(1.0, 7.0)[::2] if use_view else np.array([1.0, 3.0, 5.0])
        if readonly:
            values.setflags(write=False)
        limit = FreeJointVelocityLimit(
            self.model,
            max_linear_velocity=values if linear else None,
            max_angular_velocity=None if linear else values,
        )
        bound = limit.linear_max if linear else limit.angular_max
        assert bound is not None
        self.assertEqual(values.flags.writeable, not readonly)
        self.assertFalse(np.shares_memory(values, bound))
        self.assertFalse(bound.flags.writeable)
        np.testing.assert_array_equal(bound, [1.0, 3.0, 5.0])

    @parameterized.parameters(False, True)
    def test_qp_bounds_ignore_external_mutation(self, readonly):
        storage = np.arange(1.0, 7.0)
        linear, angular = storage[::2], storage[1::2]
        if readonly:
            linear.setflags(write=False)
            angular.setflags(write=False)
        limit = FreeJointVelocityLimit(self.model, linear, angular)
        G_before, h_before = _inequalities(limit, self.configuration, dt=0.1)

        # A read-only view does not stop its owner from modifying shared storage.
        storage[:] = 100.0

        G_after, h_after = _inequalities(limit, self.configuration, dt=0.1)
        expected_h = 0.1 * np.tile([1.0, 3.0, 5.0, 2.0, 4.0, 6.0], 2)
        np.testing.assert_array_equal(h_before, expected_h)
        np.testing.assert_array_equal(h_after, expected_h)
        np.testing.assert_array_equal(G_before, np.vstack([np.eye(6), -np.eye(6)]))
        np.testing.assert_array_equal(G_after, G_before)

    @parameterized.named_parameters(
        ("scalar", 2.0),
        ("scalar_array", np.array(2.0)),
        ("list", [2.0, 2.0, 2.0]),
        ("integer_array", np.array([2, 2, 2])),
    )
    def test_scalar_and_sequence_inputs(self, values):
        limit = FreeJointVelocityLimit(self.model, values, values)
        for bound in (limit.linear_max, limit.angular_max):
            assert bound is not None
            np.testing.assert_array_equal(bound, [2.0, 2.0, 2.0])
            self.assertFalse(bound.flags.writeable)
        _, h = _inequalities(limit, self.configuration, dt=0.1)
        np.testing.assert_array_equal(h, np.full(12, 0.2))


class TestFreeJointVelocityLimit(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("g1_mj_description")

    def setUp(self):
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("stand")

    def test_dimensions(self):
        limit = FreeJointVelocityLimit(self.model, 1.0, 2.0)
        G, h = _inequalities(limit, self.configuration, 1e-2)
        self.assertEqual(G.shape, (12, self.model.nv))
        self.assertEqual(h.shape, (12,))

    def test_linear_only_and_angular_only_shapes(self):
        lin = FreeJointVelocityLimit(self.model, max_linear_velocity=1.0)
        ang = FreeJointVelocityLimit(self.model, max_angular_velocity=1.0)
        G_lin, _ = _inequalities(lin, self.configuration, 1e-2)
        G_ang, _ = _inequalities(ang, self.configuration, 1e-2)
        self.assertEqual(G_lin.shape, (6, self.model.nv))
        self.assertEqual(G_ang.shape, (6, self.model.nv))

    def test_scalar_matches_vector(self):
        scalar = FreeJointVelocityLimit(self.model, 1.0, 2.0)
        vector = FreeJointVelocityLimit(self.model, [1.0] * 3, [2.0] * 3)
        _, h_s = _inequalities(scalar, self.configuration, 1e-2)
        _, h_v = _inequalities(vector, self.configuration, 1e-2)
        np.testing.assert_allclose(h_s, h_v)

    def test_rhs_scales_with_dt(self):
        limit = FreeJointVelocityLimit(self.model, 1.0, 2.0)
        _, h1 = _inequalities(limit, self.configuration, dt=0.1)
        _, h2 = _inequalities(limit, self.configuration, dt=0.2)
        np.testing.assert_allclose(h2, 2.0 * h1)

    def test_angular_block_is_body_local(self):
        """Angular rows are identity on the angular DOFs, orientation-independent."""
        limit = FreeJointVelocityLimit(self.model, max_angular_velocity=1.0)
        _rotated(self.configuration, [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        G, _ = _inequalities(limit, self.configuration, 1e-2)
        np.testing.assert_allclose(G[:3, 3:6], np.eye(3))

    def test_linear_block_rotates_with_base(self):
        """Linear rows equal the base-from-world rotation, not the identity."""
        limit = FreeJointVelocityLimit(self.model, max_linear_velocity=1.0)
        _rotated(self.configuration, [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        R = self.configuration.data.xmat[limit.base_body_id].reshape(3, 3)
        G, _ = _inequalities(limit, self.configuration, 1e-2)
        np.testing.assert_allclose(G[:3, :3], R.T)
        self.assertFalse(np.allclose(G[:3, :3], np.eye(3)))

    def test_requires_at_least_one_bound(self):
        with self.assertRaises(LimitDefinitionError):
            FreeJointVelocityLimit(self.model)

    def test_invalid_shape(self):
        with self.assertRaises(LimitDefinitionError):
            FreeJointVelocityLimit(self.model, max_linear_velocity=[1.0, 2.0])

    def test_no_free_joint(self):
        model = mujoco.MjModel.from_xml_string(_NO_FREE)
        with self.assertRaises(LimitDefinitionError):
            FreeJointVelocityLimit(model, 1.0)

    def test_multiple_free_joints_requires_name(self):
        model = mujoco.MjModel.from_xml_string(_TWO_FREE)
        with self.assertRaises(LimitDefinitionError):
            FreeJointVelocityLimit(model, 1.0)
        # Naming one resolves the ambiguity and targets its DOFs.
        limit = FreeJointVelocityLimit(model, 1.0, joint_name="fb")
        self.assertEqual(limit.base_body_id, model.body("b").id)

    def test_named_joint_must_be_free(self):
        model = mujoco.MjModel.from_xml_string(_NO_FREE)
        with self.assertRaises(LimitDefinitionError):
            FreeJointVelocityLimit(model, 1.0, joint_name="h")

    def test_solve_respects_limit(self):
        """The #162 regression: base twist stays within the base-frame caps."""
        task = FrameTask("left_wrist_yaw_link", "body", 1.0, 1.0)
        task.set_target(SE3.from_translation(np.array([10.0, 10.0, 10.0])))
        limit = FreeJointVelocityLimit(self.model, 1.0, 2.0)
        configuration = Configuration(self.model)
        configuration.update_from_keyframe("stand")
        _rotated(configuration, [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        for _ in range(8):
            v = mink.solve_ik(configuration, [task], 1e-2, "daqp", limits=[limit])
            R = configuration.data.xmat[limit.base_body_id].reshape(3, 3)
            self.assertLessEqual(np.abs(R.T @ v[:3]).max(), 1.0 + 1e-6)
            self.assertLessEqual(np.abs(v[3:6]).max(), 2.0 + 1e-6)
            configuration.integrate_inplace(v, 1e-2)

    def test_composes_with_velocity_limit(self):
        """Base twist and joint velocities respect their separate caps."""
        task = FrameTask("left_wrist_yaw_link", "body", 1.0, 1.0)
        task.set_target(SE3.from_translation(np.array([10.0, 10.0, 10.0])))
        base = FreeJointVelocityLimit(self.model, 1.0, 2.0)
        joints = VelocityLimit(
            self.model,
            {self.model.joint(i).name: np.pi for i in range(1, self.model.njnt)},
        )
        configuration = Configuration(self.model)
        configuration.update_from_keyframe("stand")
        for _ in range(8):
            v = mink.solve_ik(
                configuration, [task], 1e-2, "daqp", limits=[base, joints]
            )
            R = configuration.data.xmat[base.base_body_id].reshape(3, 3)
            self.assertLessEqual(np.abs(R.T @ v[:3]).max(), 1.0 + 1e-6)
            self.assertLessEqual(np.abs(v[3:6]).max(), 2.0 + 1e-6)
            self.assertLessEqual(np.abs(v[joints.indices]).max(), np.pi + 1e-6)
            configuration.integrate_inplace(v, 1e-2)


if __name__ == "__main__":
    absltest.main()
