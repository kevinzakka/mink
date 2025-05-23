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

    def test_exceeding_limits_with_safety_break_throws(self):
        """IK checks for configuration limits if safety_break is True."""
        q = self.model.key("home").qpos.copy()
        q[0] = self.model.jnt_range[0, 1] + 0.1
        self.configuration.update(q)
        with self.assertRaises(mink.NotWithinConfigurationLimits):
            mink.solve_ik(
                self.configuration,
                [],
                limits=self.limits,
                dt=1.0,
                safety_break=True,
                solver="daqp",
            )

    def test_exceeding_limits_without_safety_break_does_not_throw(self):
        """IK ignores configuration limits if safety_break is False."""
        q = self.model.key("home").qpos.copy()
        q[0] = self.model.jnt_range[0, 1] + 0.1
        self.configuration.update(q)
        mink.solve_ik(
            self.configuration,
            [],
            limits=self.limits,
            dt=1.0,
            solver="daqp",
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
        v = mink.solve_ik(self.configuration, [], limits=[], dt=1e-3, solver="daqp")
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
            self.configuration, [task], limits=self.limits, dt=1e-3, solver="daqp"
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
        velocity_tol = 1e-4  # [m/s]
        velocity = mink.solve_ik(
            configuration, [task], limits=self.limits, dt=dt, solver="daqp"
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
            if error < 1e-6 and np.allclose(velocity, 0.0, atol=velocity_tol):
                break
            self.assertLess(error, last_error)  # Error stictly decreases.
            last_error = error
            configuration.integrate_inplace(velocity, dt)
            velocity = mink.solve_ik(
                configuration, [task], limits=self.limits, dt=dt, solver="daqp"
            )

        # After nb_steps we are at the target and not moving.
        self.assertTrue(np.allclose(velocity, 0.0, atol=velocity_tol))
        self.assertAlmostEqual(norm(task.compute_error(configuration)), 0.0, places=5)
        np.testing.assert_allclose(
            configuration.get_transform_frame_to_world(
                "attachment_site", "site"
            ).as_matrix(),
            transform_target_to_world.as_matrix(),
            atol=1e-6,
        )
        self.assertLess(nb_steps, 20)

    def test_no_solution_found_throws(self):
        """When the QP solver fails to find a solution, an exception is raised."""
        # Ask the end-effector to move to a far away target with a very large cost.
        task = mink.FrameTask("attachment_site", "site", 1e6, 0)
        task.set_target(mink.SE3.from_translation(np.array([100.0, 0, 0])))
        with self.assertRaises(mink.NoSolutionFound) as cm:
            mink.solve_ik(self.configuration, [task], dt=1e-3, solver="daqp")
        self.assertEqual(str(cm.exception), "QP solver daqp failed to find a solution.")


if __name__ == "__main__":
    absltest.main()
