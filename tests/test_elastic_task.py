"""Tests for elastic (L1-penalized) tasks."""

import numpy as np
import qpsolvers
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

import mink
from mink.solve_ik import (
    _compute_qp_equalities,
    _compute_qp_inequalities,
    _compute_qp_objective,
)

_SOLVER = "daqp"
_SITE = "attachment_site"


class TestElasticTask(absltest.TestCase):
    """Tests for tasks with `elastic=True`."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("ur5e_mj_description")
        cls.nv = cls.model.nv

    def setUp(self):
        self.configuration = mink.Configuration(self.model)
        self.configuration.update_from_keyframe("home")
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
        # Posture regularizer pulling away from the current configuration.
        self.posture_task = mink.PostureTask(self.model, cost=1.0)
        self.posture_task.set_target(self.configuration.q + 0.3)

    def _frame_task(self, offset=(0.05, -0.03, 0.04), rotation=None, **kwargs):
        """Frame task whose target is the current site pose moved by `offset`."""
        task = mink.FrameTask(
            _SITE, "site", position_cost=1.0, orientation_cost=1.0, **kwargs
        )
        transform = mink.SE3.from_translation(np.asarray(offset))
        if rotation is not None:
            transform = transform @ mink.SE3.from_rotation(rotation)
        init = self.configuration.get_transform_frame_to_world(_SITE, "site")
        task.set_target(init @ transform)
        return task

    def _solve(self, problem: qpsolvers.Problem) -> np.ndarray:
        """Solve the QP, returning the full solution [dq; s]."""
        result = qpsolvers.solve_problem(problem, solver=_SOLVER)
        self.assertTrue(result.found)
        assert result.x is not None
        return result.x

    def _linearized_residual(self, task: mink.Task, delta_q: np.ndarray):
        """Residual J dq + alpha e of the task's first-order dynamics."""
        return task.compute_jacobian(
            self.configuration
        ) @ delta_q + task.gain * task.compute_error(self.configuration)

    # Problem structure.

    def test_no_elastic_problem_is_unchanged(self):
        """Without elastic tasks, build_ik is exactly the plain QP of the tasks."""
        frame_task = self._frame_task()
        freeze_task = mink.DofFreezingTask(self.model, dof_indices=[0])
        tasks = [frame_task, self.posture_task]
        problem = mink.build_ik(
            self.configuration,
            tasks,
            dt=1e-2,
            limits=self.limits,
            constraints=[freeze_task],
        )
        H, c = _compute_qp_objective(self.configuration, tasks, 1e-12)
        G, h = _compute_qp_inequalities(self.configuration, self.limits, 1e-2)
        A, b = _compute_qp_equalities(self.configuration, [freeze_task])
        for actual, expected in (
            (problem.P, H),
            (problem.q, c),
            (problem.G, G),
            (problem.h, h),
            (problem.A, A),
            (problem.b, b),
        ):
            assert actual is not None and expected is not None
            self.assertEqual(actual.shape, expected.shape)
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(problem.P.shape, (self.nv, self.nv))

    def test_problem_shape_without_limits(self):
        task = self._frame_task(elastic=True)
        problem = mink.build_ik(self.configuration, [task], dt=1e-2, limits=[])
        n = self.nv + 6
        self.assertEqual(problem.P.shape, (n, n))
        self.assertEqual(problem.q.shape, (n,))
        assert problem.G is not None and problem.h is not None
        self.assertEqual(problem.G.shape, (12, n))
        self.assertEqual(problem.h.shape, (12,))
        self.assertIsNone(problem.A)
        # The slack block of the Hessian is zero and each slack costs one unit.
        np.testing.assert_array_equal(problem.P[self.nv :], 0.0)
        np.testing.assert_array_equal(problem.P[:, self.nv :], 0.0)
        np.testing.assert_array_equal(problem.q[self.nv :], 1.0)

    def test_problem_shape_with_default_limits(self):
        task = self._frame_task(elastic=True)
        plain = mink.build_ik(self.configuration, [], dt=1e-2)
        problem = mink.build_ik(self.configuration, [task], dt=1e-2)
        G_plain, h_plain, G, h = plain.G, plain.h, problem.G, problem.h
        assert isinstance(G_plain, np.ndarray) and h_plain is not None
        assert isinstance(G, np.ndarray) and h is not None
        n_limit = G_plain.shape[0]
        self.assertEqual(G.shape, (n_limit + 12, self.nv + 6))
        np.testing.assert_array_equal(G[:n_limit, : self.nv], G_plain)
        np.testing.assert_array_equal(G[:n_limit, self.nv :], 0.0)
        np.testing.assert_array_equal(h[:n_limit], h_plain)

    def test_problem_shape_with_hard_constraint(self):
        task = self._frame_task(elastic=True)
        freeze_task = mink.DofFreezingTask(self.model, dof_indices=[0, 1])
        problem = mink.build_ik(
            self.configuration, [task], dt=1e-2, constraints=[freeze_task]
        )
        assert problem.A is not None and problem.b is not None
        self.assertEqual(problem.A.shape, (2, self.nv + 6))
        np.testing.assert_array_equal(problem.A[:, self.nv :], 0.0)
        self.assertEqual(problem.b.shape, (2,))

    def test_elastic_task_is_excluded_from_objective(self):
        task = self._frame_task(elastic=True)
        damping = 1e-3
        problem = mink.build_ik(
            self.configuration, [task], dt=1e-2, damping=damping, limits=[]
        )
        np.testing.assert_array_equal(
            problem.P[: self.nv, : self.nv], damping * np.eye(self.nv)
        )
        np.testing.assert_array_equal(problem.q[: self.nv], 0.0)

    def test_slack_columns_of_multiple_elastic_tasks(self):
        """Each elastic task's -I block sits only in its own slack columns, with the
        rows scaled by that task's penalty."""
        frame_penalty = np.arange(1.0, 7.0)
        frame_task = self._frame_task(elastic=True, penalty=frame_penalty)
        com_task = mink.ComTask(cost=1.0, elastic=True, penalty=7.0)
        com_task.set_target_from_configuration(self.configuration)
        problem = mink.build_ik(
            self.configuration, [frame_task, com_task], dt=1e-2, limits=[]
        )
        nv = self.nv
        m = 9
        G, h = problem.G, problem.h
        assert G is not None and h is not None
        self.assertEqual(G.shape, (2 * m, nv + m))
        np.testing.assert_array_equal(G[:m, nv:], -np.eye(m))
        np.testing.assert_array_equal(G[m:, nv:], -np.eye(m))

        frame_residual = frame_task.compute_qp_residual(self.configuration)
        com_residual = com_task.compute_qp_residual(self.configuration)
        np.testing.assert_allclose(
            G[:6, :nv], frame_penalty[:, None] * frame_residual[0]
        )
        np.testing.assert_allclose(G[6:9, :nv], 7.0 * com_residual[0])
        np.testing.assert_allclose(G[m:, :nv], -G[:m, :nv])
        np.testing.assert_allclose(h[:6], frame_penalty * frame_residual[1])
        np.testing.assert_allclose(h[6:9], 7.0 * com_residual[1])
        np.testing.assert_allclose(h[m:], -h[:m])

    # Solutions.

    def test_feasible_target_matches_hard_constraint(self):
        """A feasible elastic task gives the same step as the hard constraint."""
        dt = 1e-2
        hard = mink.solve_ik(
            self.configuration,
            [self.posture_task],
            dt,
            _SOLVER,
            constraints=[self._frame_task()],
        )
        elastic_task = self._frame_task(elastic=True)
        elastic = mink.solve_ik(
            self.configuration, [elastic_task, self.posture_task], dt, _SOLVER
        )
        self.assertEqual(elastic.shape, (self.nv,))
        np.testing.assert_allclose(elastic, hard, atol=1e-8)

        problem = mink.build_ik(
            self.configuration, [elastic_task, self.posture_task], dt
        )
        x = self._solve(problem)
        np.testing.assert_allclose(x[self.nv :], 0.0, atol=1e-8)

    def test_elastic_task_beats_soft_objective(self):
        """With posture pulling away, the elastic task is met exactly while the
        quadratic task settles for a compromise."""
        dt = 1e-2
        soft_task = self._frame_task()
        elastic_task = self._frame_task(elastic=True)
        v_soft = mink.solve_ik(
            self.configuration, [soft_task, self.posture_task], dt, _SOLVER
        )
        v_elastic = mink.solve_ik(
            self.configuration, [elastic_task, self.posture_task], dt, _SOLVER
        )
        soft_residual = self._linearized_residual(soft_task, v_soft * dt)
        elastic_residual = self._linearized_residual(elastic_task, v_elastic * dt)
        self.assertGreater(np.linalg.norm(soft_residual), 1e-2)
        self.assertLess(np.linalg.norm(elastic_residual), 1e-8)

    def test_exact_penalty_threshold(self):
        """The task holds iff the penalty exceeds the hard-constraint multipliers."""
        dt = 1e-2
        hard_problem = mink.build_ik(
            self.configuration,
            [self.posture_task],
            dt,
            limits=[],
            constraints=[self._frame_task()],
        )
        hard = qpsolvers.solve_problem(hard_problem, solver=_SOLVER)
        assert hard.found and hard.x is not None and hard.y is not None
        # With unit costs, weighted and unweighted multipliers coincide.
        multipliers = np.abs(hard.y)
        largest = int(np.argmax(multipliers))
        self.assertGreater(multipliers[largest], 1e-3)

        # Penalty above every multiplier: identical to the hard constraint.
        rho = 1.05 * multipliers[largest]
        problem = mink.build_ik(
            self.configuration,
            [self._frame_task(elastic=True, penalty=rho), self.posture_task],
            dt,
            limits=[],
        )
        x = self._solve(problem)
        np.testing.assert_allclose(x[: self.nv], hard.x, atol=1e-8)
        np.testing.assert_allclose(x[self.nv :], 0.0, atol=1e-8)

        # Penalty below the largest multiplier: that component yields.
        rho = 0.7 * multipliers[largest]
        problem = mink.build_ik(
            self.configuration,
            [self._frame_task(elastic=True, penalty=rho), self.posture_task],
            dt,
            limits=[],
        )
        x = self._solve(problem)
        slack = x[self.nv :]
        self.assertGreater(slack[largest], 1e-6)
        self.assertGreater(np.abs(x[: self.nv] - hard.x).max(), 1e-6)

    def test_infeasible_hard_constraint_still_solves(self):
        """Where the hard constraint is infeasible, the elastic task yields and all
        limits hold."""
        dt = 1e-3
        offset = (0.0, 0.0, 0.2)  # Far beyond a single velocity-limited step.
        with self.assertRaises(mink.NoSolutionFound):
            mink.solve_ik(
                self.configuration,
                [self.posture_task],
                dt,
                _SOLVER,
                limits=self.limits,
                constraints=[self._frame_task(offset)],
            )

        task = self._frame_task(offset, elastic=True)
        v = mink.solve_ik(
            self.configuration,
            [task, self.posture_task],
            dt,
            _SOLVER,
            limits=self.limits,
        )
        G, h = _compute_qp_inequalities(self.configuration, self.limits, dt)
        assert G is not None and h is not None
        self.assertTrue(np.all(G @ (v * dt) <= h + 1e-9))
        # The task still makes progress toward its target.
        residual = self._linearized_residual(task, v * dt)
        self.assertLess(
            np.linalg.norm(residual),
            np.linalg.norm(task.compute_error(self.configuration)),
        )

    def test_yielding_is_sparse(self):
        """With only three free DOFs, a strong position penalty and a weak
        orientation penalty, position is held exactly and orientation yields."""
        freeze_task = mink.DofFreezingTask(self.model, dof_indices=[3, 4, 5])
        task = self._frame_task(
            offset=(0.03, 0.02, -0.02),
            rotation=mink.SO3.from_rpy_radians(0.2, -0.1, 0.3),
            elastic=True,
            penalty=[1e3, 1e3, 1e3, 1.0, 1.0, 1.0],
        )
        posture_task = mink.PostureTask(self.model, cost=1e-2)
        posture_task.set_target_from_configuration(self.configuration)
        problem = mink.build_ik(
            self.configuration,
            [task, posture_task],
            dt=1e-2,
            limits=[],
            constraints=[freeze_task],
        )
        x = self._solve(problem)
        residual = self._linearized_residual(task, x[: self.nv])
        np.testing.assert_allclose(residual[:3], 0.0, atol=1e-8)
        self.assertGreater(np.linalg.norm(residual[3:]), 1e-2)
        # Slacks are the penalty paid per component, rho * |residual|.
        slack = x[self.nv :]
        np.testing.assert_allclose(slack[:3] / 1e3, 0.0, atol=1e-8)
        np.testing.assert_allclose(slack[3:], np.abs(residual[3:]), atol=1e-8)

    def test_zero_cost_component_is_unconstrained(self):
        task = mink.FrameTask(
            _SITE,
            "site",
            position_cost=1.0,
            orientation_cost=0.0,
            elastic=True,
        )
        init = self.configuration.get_transform_frame_to_world(_SITE, "site")
        task.set_target(
            init
            @ mink.SE3.from_rotation_and_translation(
                mink.SO3.from_z_radians(0.5), np.array([0.05, 0.0, 0.0])
            )
        )
        v = mink.solve_ik(self.configuration, [task, self.posture_task], 1e-2, _SOLVER)
        residual = self._linearized_residual(task, v * 1e-2)
        np.testing.assert_allclose(residual[:3], 0.0, atol=1e-8)
        self.assertGreater(np.linalg.norm(residual[3:]), 1e-2)

    def test_multiple_elastic_tasks_yield_independently(self):
        """A frame task using all six DOFs holds; a cheap CoM task yields."""
        frame_task = self._frame_task(elastic=True)
        com_task = mink.ComTask(cost=1.0, elastic=True, penalty=1e-3)
        com_task.set_target(
            self.configuration.data.subtree_com[1] + np.array([0.0, 0.05, 0.0])
        )
        problem = mink.build_ik(
            self.configuration,
            [frame_task, com_task, self.posture_task],
            dt=1e-2,
            limits=[],
        )
        x = self._solve(problem)
        np.testing.assert_allclose(x[self.nv : self.nv + 6], 0.0, atol=1e-8)
        self.assertGreater(x[self.nv + 6 :].max(), 1e-6)
        frame_residual = self._linearized_residual(frame_task, x[: self.nv])
        np.testing.assert_allclose(frame_residual, 0.0, atol=1e-8)

    def test_closed_loop_convergence(self):
        """Integrating toward a reachable target, the elastic task converges while
        the quadratic task keeps a bias from the posture task."""
        dt = 1e-2
        offset = (0.1, -0.05, 0.05)
        rotation = mink.SO3.from_rpy_radians(0.1, 0.0, -0.2)
        posture_task = mink.PostureTask(self.model, cost=1e-1)
        posture_task.set_target_from_configuration(self.configuration)

        errors = {}
        for elastic in (False, True):
            configuration = mink.Configuration(self.model)
            configuration.update_from_keyframe("home")
            task = self._frame_task(offset, rotation, elastic=elastic)
            for _ in range(50):
                v = mink.solve_ik(
                    configuration, [task, posture_task], dt, _SOLVER, limits=self.limits
                )
                configuration.integrate_inplace(v, dt)
            errors[elastic] = np.linalg.norm(task.compute_error(configuration))

        self.assertLess(errors[True], 1e-8)
        self.assertGreater(errors[False], 1e-3)

    def test_elastic_task_in_solve_ik_with_penalty_mismatch_raises(self):
        """A per-component penalty that no longer matches the task raises at solve
        time."""
        task = self._frame_task(elastic=True)
        task.penalty = np.ones(3)
        with self.assertRaises(mink.InvalidPenalty):
            mink.build_ik(self.configuration, [task], dt=1e-2)

    # Forwarded constructor arguments.

    def test_pose_tasks_forward_elastic_arguments(self):
        target = self.configuration.get_transform_frame_to_world(_SITE, "site")
        target = target @ mink.SE3.from_translation(np.array([0.02, 0.0, 0.0]))
        root_to_world = self.configuration.get_transform_frame_to_world("base", "body")

        frame_task = mink.FrameTask(
            _SITE, "site", 1.0, 1.0, elastic=True, penalty=np.full(6, 2.0)
        )
        frame_task.set_target(target)
        relative_task = mink.RelativeFrameTask(
            _SITE,
            "site",
            "base",
            "body",
            1.0,
            1.0,
            elastic=True,
            penalty=np.full(6, 2.0),
        )
        relative_task.set_target(root_to_world.inverse() @ target)
        com_task = mink.ComTask(cost=1.0, elastic=True, penalty=np.full(3, 2.0))
        com_task.set_target_from_configuration(self.configuration)
        look_at_task = mink.LookAtTask(
            _SITE, "site", cost=1.0, elastic=True, penalty=np.full(3, 2.0)
        )
        look_at_task.set_target(target.translation() + np.array([0.0, 0.0, -1.0]))
        axis_align_task = mink.AxisAlignTask(
            _SITE, "site", cost=1.0, elastic=True, penalty=np.full(3, 2.0)
        )
        axis_align_task.set_target(np.array([0.0, 0.1, -1.0]))

        for task in (
            frame_task,
            relative_task,
            com_task,
            look_at_task,
            axis_align_task,
        ):
            with self.subTest(task=type(task).__name__):
                self.assertTrue(task.elastic)
                k = task.cost.shape[0]
                np.testing.assert_array_equal(task._penalty_vector(k), np.full(k, 2.0))
                problem = mink.build_ik(
                    self.configuration, [task, self.posture_task], dt=1e-2
                )
                self.assertEqual(problem.P.shape, (self.nv + k, self.nv + k))
                self._solve(problem)

    def test_pose_tasks_reject_elastic_with_lm_damping(self):
        constructors = (
            lambda: mink.FrameTask(
                _SITE, "site", 1.0, 1.0, lm_damping=1.0, elastic=True
            ),
            lambda: mink.RelativeFrameTask(
                _SITE, "site", "base", "body", 1.0, 1.0, lm_damping=1.0, elastic=True
            ),
            lambda: mink.ComTask(cost=1.0, lm_damping=1.0, elastic=True),
            lambda: mink.LookAtTask(_SITE, "site", lm_damping=1.0, elastic=True),
            lambda: mink.AxisAlignTask(_SITE, "site", lm_damping=1.0, elastic=True),
        )
        for constructor in constructors:
            with self.assertRaises(mink.InvalidDamping):
                constructor()

    def test_pose_tasks_reject_invalid_penalty(self):
        with self.assertRaises(mink.InvalidPenalty):
            mink.FrameTask(_SITE, "site", 1.0, 1.0, elastic=True, penalty=np.ones(3))
        with self.assertRaises(mink.InvalidPenalty):
            mink.ComTask(cost=1.0, elastic=True, penalty=-1.0)


if __name__ == "__main__":
    absltest.main()
