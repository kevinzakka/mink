"""Tests for solve_ik.py."""

from typing import cast

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

    def test_fused_objective_matches_per_task_sum(self):
        """The fused QP objective equals the naive per-task sum, across a frame,
        posture, and relative-frame task (all fused) plus a task that uses the
        dense fallback (KineticEnergyRegularizationTask, whose Hessian is
        inertia-weighted rather than J^T J)."""
        self.configuration.update_from_keyframe("home")
        frame_task = mink.FrameTask(
            "attachment_site",
            "site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1e-2,
        )
        frame_task.set_target_from_configuration(self.configuration)
        posture_task = mink.PostureTask(self.model, cost=1e-1, lm_damping=1e-3)
        posture_task.set_target_from_configuration(self.configuration)
        relative_task = mink.RelativeFrameTask(
            "attachment_site",
            "site",
            root_name="upper_arm_link",
            root_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1e-3,
        )
        relative_task.set_target_from_configuration(self.configuration)
        ke_task = mink.KineticEnergyRegularizationTask(cost=1e-4)
        ke_task.set_dt(0.02)
        tasks = [frame_task, posture_task, relative_task, ke_task]

        # Perturb so the task errors (and hence the LM terms) are non-zero.
        q = self.model.key("home").qpos.copy()
        q[:3] += 0.2
        self.configuration.update(q)

        damping = 1e-6
        problem = mink.build_ik(self.configuration, tasks, dt=0.02, damping=damping)

        nv = self.model.nv
        H = np.eye(nv) * damping
        c = np.zeros(nv)
        for task in tasks:
            obj = task.compute_qp_objective(self.configuration)
            H += obj.H
            c += obj.c

        # P is typed ndarray|csc_matrix but is dense here; q is already ndarray.
        P = cast(np.ndarray, problem.P)
        np.testing.assert_allclose(P, H, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(problem.q, c, rtol=1e-9, atol=1e-12)

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
        self.assertAlmostEqual(float(norm(task.compute_error(configuration))), 0.1)
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(
                configuration.get_transform_frame_to_world(
                    "attachment_site", "site"
                ).as_matrix(),
                transform_target_to_world.as_matrix(),
            )

        last_error = 1e6
        nb_steps = 0
        for _ in range(50):
            error = norm(task.compute_error(configuration))
            if error < 1e-6 and np.allclose(velocity, 0.0, atol=velocity_tol):
                break
            self.assertLess(error, last_error)  # Error stictly decreases.
            last_error = error
            configuration.integrate_inplace(velocity, dt)
            velocity = mink.solve_ik(
                configuration, [task], limits=self.limits, dt=dt, solver="daqp"
            )
            nb_steps += 1

        # After nb_steps we are at the target and not moving.
        self.assertTrue(np.allclose(velocity, 0.0, atol=velocity_tol))
        self.assertAlmostEqual(
            float(norm(task.compute_error(configuration))), 0.0, places=5
        )
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

    def test_equality_constraints_freeze_dofs(self):
        """DOF freezing via equality constraints keeps specified DOFs constant."""
        configuration = mink.Configuration(self.model)
        configuration.update_from_keyframe("home")

        # Store initial configuration.
        q_init = configuration.q.copy()

        # Create a task to move the end effector.
        task = mink.FrameTask(
            "attachment_site", "site", position_cost=1.0, orientation_cost=1.0
        )
        transform_init_to_world = configuration.get_transform_frame_to_world(
            "attachment_site",
            "site",
        )
        transform_target_to_init = mink.SE3.from_translation(np.array([0, 0, 0.05]))
        transform_target_to_world = transform_init_to_world @ transform_target_to_init
        task.set_target(transform_target_to_world)

        # Create DOF freezing constraint for first two joints.
        frozen_dofs = [0, 1]
        freeze_task = mink.DofFreezingTask(model=self.model, dof_indices=frozen_dofs)

        dt = 5e-3  # [s]

        # Solve IK with equality constraint.
        for _ in range(20):
            velocity = mink.solve_ik(
                configuration,
                [task],
                constraints=[freeze_task],
                limits=self.limits,
                dt=dt,
                solver="daqp",  # DAQP supports equality constraints.
            )
            configuration.integrate_inplace(velocity, dt)

        # Check that frozen DOFs haven't changed.
        for dof in frozen_dofs:
            self.assertAlmostEqual(
                configuration.q[dof],
                q_init[dof],
                places=10,
                msg=f"DOF {dof} should remain frozen",
            )

        # Check that at least some other DOFs have changed.
        other_dofs = [i for i in range(self.model.nv) if i not in frozen_dofs]
        dof_changes = [abs(configuration.q[i] - q_init[i]) for i in other_dofs]
        self.assertGreater(
            max(dof_changes),
            1e-3,
            msg="At least one non-frozen DOF should have moved",
        )

    def test_equality_constraints_none_by_default(self):
        """By default, no equality constraints are set."""
        problem = mink.build_ik(self.configuration, [], dt=1.0)
        self.assertIsNone(problem.A)
        self.assertIsNone(problem.b)

    def test_equality_constraints_set_when_provided(self):
        """When constraints are provided, A and b matrices are set."""
        freeze_task = mink.DofFreezingTask(model=self.model, dof_indices=[0, 1])
        problem = mink.build_ik(
            self.configuration, [], constraints=[freeze_task], dt=1.0
        )
        self.assertIsNotNone(problem.A)
        self.assertIsNotNone(problem.b)
        assert problem.A is not None  # Type narrowing for type checker
        assert problem.b is not None  # Type narrowing for type checker
        self.assertEqual(problem.A.shape, (2, self.model.nv))
        self.assertEqual(problem.b.shape, (2,))


if __name__ == "__main__":
    absltest.main()
