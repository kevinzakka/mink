"""Tests for elastic.py."""

import importlib
from unittest import mock

import mujoco
import numpy as np
import qpsolvers
from absl.testing import absltest

import mink

solve_ik_module = importlib.import_module("mink.solve_ik")

# 6-dof floating tool plus a redundant roll joint about the tool z axis.
_XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <joint name="tx" type="slide" axis="1 0 0" range="-.5 .5"/>
      <joint name="ty" type="slide" axis="0 1 0" range="-.5 .5"/>
      <joint name="tz" type="slide" axis="0 0 1" range="-.5 .5"/>
      <joint name="rx" type="hinge" axis="1 0 0" range="-2 2"/>
      <joint name="ry" type="hinge" axis="0 1 0" range="-2 2"/>
      <joint name="rz" type="hinge" axis="0 0 1" range="-2 2"/>
      <geom size=".05"/>
      <body name="tool">
        <joint name="roll" type="hinge" axis="0 0 1" range="-2 2"/>
        <geom size=".02"/>
        <site name="tip"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _setup(target_xyz):
    model = mujoco.MjModel.from_xml_string(_XML)
    configuration = mink.Configuration(model)
    frame_task = mink.FrameTask("tip", "site", 1.0, 1.0)
    frame_task.set_target(mink.SE3.from_translation(np.asarray(target_xyz)))
    posture_task = mink.PostureTask(model, cost=0.1)
    posture_task.set_target(np.array([0.1, -0.2, 0.1, 0.3, -0.2, 0.4, 0.5]))
    return configuration, frame_task, posture_task


def _residual(configuration, task, v, dt):
    return task.compute_jacobian(configuration) @ v * dt + task.compute_error(
        configuration
    )


class TestElastic(absltest.TestCase):
    def test_held_step_is_the_hard_solution(self):
        configuration, frame_task, posture_task = _setup([0.1, 0.2, -0.1])
        frame_task.gain = 0.3
        v_hard = mink.solve_ik(
            configuration, [posture_task], 0.01, "daqp", constraints=[frame_task]
        )
        with mock.patch.object(
            qpsolvers, "solve_problem", wraps=qpsolvers.solve_problem
        ) as solve:
            v = mink.solve_ik(
                configuration,
                [posture_task],
                0.01,
                "daqp",
                constraints=[mink.Elastic(frame_task)],
            )
        self.assertEqual(solve.call_count, 1)
        np.testing.assert_allclose(v, v_hard, atol=1e-10)

    def test_held_iff_penalty_exceeds_multiplier(self):
        configuration, frame_task, posture_task = _setup([0.1, 0.2, -0.1])
        posture_task.set_cost(1.0)
        problem = mink.build_ik(
            configuration, [posture_task], 0.01, limits=[], constraints=[frame_task]
        )
        result = qpsolvers.solve_problem(problem, solver="daqp")
        assert result.y is not None
        lam = np.abs(result.y)
        i = int(np.argmax(lam))

        held = mink.Elastic(frame_task, penalty=lam + 0.1)
        v = mink.solve_ik(
            configuration, [posture_task], 0.01, "daqp", limits=[], constraints=[held]
        )
        r = _residual(configuration, frame_task, v, 0.01)
        np.testing.assert_allclose(r, 0.0, atol=1e-6)

        penalty = lam + 0.1
        penalty[i] = 0.5 * lam[i]
        yielding = mink.Elastic(frame_task, penalty=penalty)
        v = mink.solve_ik(
            configuration,
            [posture_task],
            0.01,
            "daqp",
            limits=[],
            constraints=[yielding],
        )
        r = _residual(configuration, frame_task, v, 0.01)
        self.assertGreater(abs(r[i]), 1e-4)

    def test_yields_when_hard_constraint_is_infeasible(self):
        configuration, frame_task, posture_task = _setup([1.0, 0.2, -0.1])
        with self.assertRaises(mink.NoSolutionFound):
            mink.solve_ik(
                configuration, [posture_task], 0.01, "daqp", constraints=[frame_task]
            )
        v = mink.solve_ik(
            configuration,
            [posture_task],
            0.01,
            "daqp",
            constraints=[mink.Elastic(frame_task)],
        )
        r = _residual(configuration, frame_task, v, 0.01)
        self.assertGreater(abs(r[0]), 0.1)
        np.testing.assert_allclose(r[1:], 0.0, atol=1e-8)

    def test_stronger_penalty_wins_conflict(self):
        xml = """
        <mujoco>
          <worldbody>
            <body>
              <joint type="slide" axis="1 0 0"/>
              <geom size=".1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        configuration = mink.Configuration(model)

        class Pin(mink.Task):
            def __init__(self, target):
                super().__init__(cost=np.ones(1))
                self.target = target

            def compute_error(self, configuration):
                return configuration.q[:1] - self.target

            def compute_jacobian(self, configuration):
                return np.ones((1, 1))

        v = mink.solve_ik(
            configuration,
            [],
            0.01,
            "daqp",
            damping=0.1,
            limits=[],
            constraints=[
                mink.Elastic(Pin(0.1), penalty=2e3),
                mink.Elastic(Pin(-0.1), penalty=1e3),
            ],
        )
        self.assertAlmostEqual(v[0] * 0.01, 0.1, places=8)

    def test_combines_with_hard_constraints(self):
        configuration, frame_task, posture_task = _setup([0.1, 0.2, -0.1])
        freeze = mink.DofFreezingTask(configuration.model, dof_indices=[6])
        v = mink.solve_ik(
            configuration,
            [posture_task],
            0.01,
            "daqp",
            constraints=[freeze, mink.Elastic(frame_task)],
        )
        self.assertAlmostEqual(v[6], 0.0, places=8)
        r = _residual(configuration, frame_task, v, 0.01)
        np.testing.assert_allclose(r, 0.0, atol=1e-8)

    def test_zero_penalty_frees_component(self):
        configuration, frame_task, posture_task = _setup([0.1, 0.2, -0.1])
        elastic = mink.Elastic(frame_task, penalty=[1e3, 1e3, 1e3, 0.0, 0.0, 0.0])
        problem = mink.build_ik(
            configuration, [posture_task], 0.01, constraints=[elastic]
        )
        # One slack per penalized component; the zero-penalty ones are dropped.
        self.assertEqual(problem.P.shape, (configuration.nv + 3,) * 2)
        v = mink.solve_ik(
            configuration, [posture_task], 0.01, "daqp", constraints=[elastic]
        )
        r = _residual(configuration, frame_task, v, 0.01)
        np.testing.assert_allclose(r[:3], 0.0, atol=1e-8)
        self.assertGreater(np.linalg.norm(r[3:]), 1e-4)

    def test_yielding_at_singularity_does_not_chatter(self):
        xml = """
        <mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <body name="upper">
              <joint name="shoulder" type="hinge" axis="0 0 1" range="-3 3"/>
              <geom type="capsule" fromto="0 0 0 .5 0 0" size=".02"/>
              <body name="lower" pos=".5 0 0">
                <joint name="elbow" type="hinge" axis="0 0 1" range="-3 3"/>
                <geom type="capsule" fromto="0 0 0 .5 0 0" size=".02"/>
                <site name="tip" pos=".5 0 0"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        configuration = mink.Configuration(model)
        configuration.update(np.array([0.3, 0.6]))
        task = mink.FrameTask("tip", "site", 1.0, 0.0)
        task.set_target(mink.SE3.from_translation(np.array([2.0, 0.5, 0.0])))
        posture_task = mink.PostureTask(model, cost=0.1)
        posture_task.set_target(np.zeros(2))
        limits = [
            mink.ConfigurationLimit(model),
            mink.VelocityLimit(model, {"shoulder": 3.0, "elbow": 3.0}),
        ]
        constraints = [mink.Elastic(task, penalty=[1e3, 1e3, 1e3, 0.0, 0.0, 0.0])]
        velocities = []
        for _ in range(200):
            v = mink.solve_ik(
                configuration,
                [posture_task],
                0.01,
                "daqp",
                limits=limits,
                constraints=constraints,
            )
            configuration.integrate_inplace(v, 0.01)
            velocities.append(v)
        # The unreachable target stretches the arm to its elbow singularity.
        self.assertLess(abs(configuration.q[1]), 0.1)
        self.assertLess(np.abs(np.diff(velocities[-50:], axis=0)).max(), 1e-6)

    def test_number_of_qp_solves(self):
        for target, expected in (([0.1, 0.2, -0.1], 1), ([1.0, 0.2, -0.1], 2)):
            with self.subTest(target=target):
                configuration, frame_task, posture_task = _setup(target)
                with mock.patch.object(
                    qpsolvers, "solve_problem", wraps=qpsolvers.solve_problem
                ) as solve:
                    mink.solve_ik(
                        configuration,
                        [posture_task],
                        0.01,
                        "daqp",
                        constraints=[mink.Elastic(frame_task)],
                    )
                self.assertEqual(solve.call_count, expected)

    def test_penalized_qp_is_robust_to_extreme_penalties(self):
        """With the default split, the penalized QP matches a reference built with
        the split that is best conditioned for that penalty: rho on the cost for
        small penalties, rho on the rows for large ones."""
        configuration, frame_task, posture_task = _setup([0.1, 0.2, -0.1])
        nv = configuration.nv
        for rho in (1e-5, 1e-3, 1.0, 1e3, 1e7, 1e11):
            with self.subTest(rho=rho):
                constraints = [mink.Elastic(frame_task, penalty=rho)]
                problem = mink.build_ik(
                    configuration, [posture_task], 0.01, constraints=constraints
                )
                result = qpsolvers.solve_problem(problem, solver="daqp")
                assert result.x is not None
                dq = result.x[:nv]
                if rho > 1e7:
                    # No reference split solves this; the constraint must hold.
                    r = _residual(configuration, frame_task, dq / 0.01, 0.01)
                    np.testing.assert_allclose(r, 0.0, atol=1e-6)
                    continue
                split = 0.0 if rho <= 1.0 else 1.0
                with mock.patch.object(solve_ik_module, "_PENALTY_SPLIT", split):
                    reference = mink.build_ik(
                        configuration, [posture_task], 0.01, constraints=constraints
                    )
                expected = qpsolvers.solve_problem(reference, solver="daqp")
                assert expected.x is not None
                np.testing.assert_allclose(dq, expected.x[:nv], atol=1e-7)

    def test_penalized_qp_keeps_hard_constraints(self):
        configuration, frame_task, posture_task = _setup([1.0, 0.2, -0.1])
        freeze = mink.DofFreezingTask(configuration.model, dof_indices=[6])
        constraints = [freeze, mink.Elastic(frame_task)]
        problem = mink.build_ik(
            configuration, [posture_task], 0.01, constraints=constraints
        )
        assert problem.A is not None
        self.assertEqual(problem.A.shape, (1, configuration.nv + 6))
        np.testing.assert_array_equal(problem.A[:, configuration.nv :], 0.0)
        v = mink.solve_ik(
            configuration, [posture_task], 0.01, "daqp", constraints=constraints
        )
        self.assertAlmostEqual(v[6], 0.0, places=8)
        r = _residual(configuration, frame_task, v, 0.01)
        self.assertGreater(abs(r[0]), 0.1)

    def test_invalid_penalty_raises(self):
        configuration, frame_task, posture_task = _setup([0.0, 0.0, 0.0])
        for penalty in (-1.0, np.nan, np.inf, np.ones((2, 3))):
            with self.assertRaises(mink.InvalidConstraint):
                mink.Elastic(frame_task, penalty=penalty)
        elastic = mink.Elastic(frame_task, penalty=[1.0, 1.0, 1.0])
        with self.assertRaises(mink.InvalidConstraint):
            mink.build_ik(configuration, [posture_task], 0.01, constraints=[elastic])


if __name__ == "__main__":
    absltest.main()
