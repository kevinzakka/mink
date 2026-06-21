"""Tests for look_at_task.py."""

import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

import mink
from mink import Configuration
from mink.exceptions import InvalidTarget, TargetNotSet, TaskDefinitionError
from mink.tasks import LookAtTask


class TestLookAtTask(absltest.TestCase):
    """Test consistency of the look-at task."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self):
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("home")

    def test_axis_normalized(self):
        task = LookAtTask("attachment_site", "site", axis=(0.0, 0.0, 2.0))
        np.testing.assert_allclose(task.axis, [0.0, 0.0, 1.0])

    def test_axis_invalid_shape(self):
        with self.assertRaises(TaskDefinitionError):
            LookAtTask("attachment_site", "site", axis=(0.0, 1.0))

    def test_axis_zero(self):
        with self.assertRaises(TaskDefinitionError):
            LookAtTask("attachment_site", "site", axis=(0.0, 0.0, 0.0))

    def test_cost_invalid_shape(self):
        with self.assertRaises(TaskDefinitionError):
            LookAtTask("attachment_site", "site", cost=[1.0, 2.0])

    def test_cost_negative(self):
        with self.assertRaises(TaskDefinitionError):
            LookAtTask("attachment_site", "site", cost=-1.0)

    def test_error_without_target(self):
        task = LookAtTask("attachment_site", "site")
        with self.assertRaises(TargetNotSet):
            task.compute_error(self.configuration)

    def test_jacobian_without_target(self):
        task = LookAtTask("attachment_site", "site")
        with self.assertRaises(TargetNotSet):
            task.compute_jacobian(self.configuration)

    def test_target_invalid_shape(self):
        task = LookAtTask("attachment_site", "site")
        with self.assertRaises(InvalidTarget):
            task.set_target([1.0, 2.0])

    def test_set_target_from_configuration_is_noop(self):
        """A target taken from the current configuration yields zero error."""
        task = LookAtTask("attachment_site", "site", axis=(0.0, 0.0, 1.0))
        task.set_target_from_configuration(self.configuration)
        error = task.compute_error(self.configuration)
        np.testing.assert_allclose(error, np.zeros(3), atol=1e-9)

    def test_jacobian_is_rank_two(self):
        """The line-of-sight projector makes the task rank two (roll is free)."""
        task = LookAtTask("attachment_site", "site", axis=(0.0, 0.0, 1.0))
        task.set_target([0.5, 0.4, 0.6])
        jacobian = task.compute_jacobian(self.configuration)
        self.assertLessEqual(np.linalg.matrix_rank(jacobian, tol=1e-6), 2)

    def test_residual_matches_objective(self):
        """The fused residual path agrees with the dense (H, c) objective."""
        task = LookAtTask("attachment_site", "site", cost=2.0)
        task.set_target([0.5, 0.4, 0.6])
        objective = task.compute_qp_objective(self.configuration)
        W, e, mu = task.compute_qp_residual(self.configuration)
        H = W.T @ W + mu * np.eye(self.model.nv)
        c = -e @ W
        np.testing.assert_allclose(objective.H, H, atol=1e-12)
        np.testing.assert_allclose(objective.c, c, atol=1e-12)

    def test_convergence(self):
        """IK drives the gaze axis to point at the target."""
        task = LookAtTask("attachment_site", "site", axis=(0.0, 0.0, 1.0), cost=1.0)
        posture = mink.PostureTask(self.model, cost=1e-3)
        posture.set_target_from_configuration(self.configuration)

        target = np.array([0.6, 0.3, 0.7])
        task.set_target(target)

        dt = 1e-2
        for _ in range(500):
            vel = mink.solve_ik(
                self.configuration, [task, posture], dt, "daqp", damping=1e-6
            )
            self.configuration.integrate_inplace(vel, dt)

        transform = self.configuration.get_transform_frame_to_world(
            "attachment_site", "site"
        )
        gaze = transform.rotation().as_matrix() @ task.axis
        desired = target - transform.translation()
        desired /= np.linalg.norm(desired)
        angle = np.arccos(np.clip(gaze @ desired, -1.0, 1.0))
        self.assertLess(angle, np.deg2rad(1.0))


if __name__ == "__main__":
    absltest.main()
