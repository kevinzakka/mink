"""Tests for solve_ik.py."""

import numpy as np
from absl.testing import absltest
from numpy.linalg import norm
from robot_descriptions.loaders.mujoco import load_robot_description

import mink


class TestSolveIK(absltest.TestCase):
    """Tests for the `solve_ik` function."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self):
        self.configuration = mink.Configuration(self.model)
        velocities = {
            "shoulder_pan_joint": np.pi,
            "shoulder_lift_joint": np.pi,
            "elbow_joint": np.pi,
            "wrist_1_joint": np.pi,
            "wrist_2_joint": np.pi,
            "wrist_3_joint": np.pi,
        }
        self.limits = [
            mink.ConfigurationLimit(self.model),
            mink.VelocityLimit(self.model, velocities),
        ]

    def test_checks_configuration_limits(self):
        """IK checks for configuration limits."""
        q = self.model.key("home").qpos.copy()
        q[0] = np.inf
        self.configuration.update(q)
        with self.assertRaises(mink.NotWithinConfigurationLimits):
            mink.solve_ik(
                self.configuration,
                [],
                limits=self.limits,
                dt=1.0,
                safety_break=True,
                solver="quadprog",
            )

    def test_ignores_configuration_limits(self):
        """IK ignores configuration limits if flag is set."""
        q = self.model.key("home").qpos.copy()
        q[0] = np.inf
        self.configuration.update(q)
        mink.solve_ik(
            self.configuration,
            [],
            limits=self.limits,
            dt=1.0,
            solver="quadprog",
            safety_break=False,
        )

    def test_model_with_no_limits(self):
        """Model with no limits has no inequality constraints."""
        problem = mink.build_ik(self.configuration, [], limits=[], dt=1.0)
        self.assertIsNone(problem.G)
        self.assertIsNone(problem.h)

    def test_default_limits(self):
        """If no limits are provided, configuration limits are set."""
        problem = mink.build_ik(self.configuration, [], dt=1.0)
        self.assertIsNotNone(problem.G)
        self.assertIsNotNone(problem.h)

    def test_trivial_solution(self):
        """No task returns no velocity."""
        v = mink.solve_ik(self.configuration, [], limits=[], dt=1e-3, solver="quadprog")
        np.testing.assert_allclose(v, np.zeros((self.model.nv,)))

    def test_single_task_fulfilled(self):
        """Velocity is zero when the only task is already fulfilled."""
        task = mink.FrameTask(
            "attachment_site",
            "site",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        task.set_target(
            self.configuration.get_transform_frame_to_world("attachment_site", "site")
        )
        v = mink.solve_ik(
            self.configuration, [task], limits=self.limits, dt=1e-3, solver="quadprog"
        )
        np.testing.assert_allclose(v, np.zeros((self.model.nv,)), atol=1e-10)

    def test_single_task_convergence(self):
        """Integrating velocities makes a task converge to its target."""
        configuration = mink.Configuration(self.model)
        configuration.update_from_keyframe("home")

        task = mink.FrameTask(
            "attachment_site", "site", position_cost=1.0, orientation_cost=1.0
        )
        transform_init_to_world = configuration.get_transform_frame_to_world(
            "attachment_site",
            "site",
        )
        transform_target_to_init = mink.SE3.from_translation(np.array([0, 0, 0.1]))
        transform_target_to_world = transform_init_to_world @ transform_target_to_init
        task.set_target(transform_target_to_world)

        dt = 5e-3  # [s]
        velocity = mink.solve_ik(
            configuration, [task], limits=self.limits, dt=dt, solver="quadprog"
        )

        # Initially we are nowhere near the target and moving.
        self.assertFalse(np.allclose(velocity, 0.0))
        self.assertAlmostEqual(norm(task.compute_error(configuration)), 0.1)
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(
                configuration.get_transform_frame_to_world(
                    "attachment_site", "site"
                ).as_matrix(),
                transform_target_to_world.as_matrix(),
            )

        last_error = 1e6
        for nb_steps in range(50):
            error = norm(task.compute_error(configuration))
            if error < 1e-6 and np.allclose(velocity, 0.0):
                break
            self.assertLess(error, last_error)  # Error stictly decreases.
            last_error = error
            configuration.integrate_inplace(velocity, dt)
            velocity = mink.solve_ik(
                configuration, [task], limits=self.limits, dt=dt, solver="quadprog"
            )

        # After nb_steps we are at the target and not moving.
        self.assertTrue(np.allclose(velocity, 0.0))
        self.assertAlmostEqual(norm(task.compute_error(configuration)), 0.0)
        np.testing.assert_allclose(
            configuration.get_transform_frame_to_world(
                "attachment_site", "site"
            ).as_matrix(),
            transform_target_to_world.as_matrix(),
        )
        self.assertLess(nb_steps, 20)


if __name__ == "__main__":
    absltest.main()
