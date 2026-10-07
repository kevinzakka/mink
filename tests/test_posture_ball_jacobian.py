"""Posture-task derivatives for quaternion-valued ball joints."""

import mujoco
import numpy as np
from absl.testing import absltest

from mink import Configuration, PostureTask

SINGLE_BALL = """
<mujoco><worldbody><body>
  <joint name="ball" type="ball"/><geom type="sphere" size="0.1"/>
</body></worldbody></mujoco>
"""
MIXED_JOINTS = """
<mujoco><worldbody><body>
  <freejoint/><geom type="sphere" size="0.1"/>
  <body><joint name="hinge" type="hinge" axis="0 0 1"/>
    <geom type="sphere" size="0.1"/>
    <body><joint name="ball" type="ball"/>
      <geom type="sphere" size="0.1"/>
      <body><joint name="slide" type="slide" axis="1 0 0"/>
        <geom type="sphere" size="0.1"/>
      </body>
    </body>
  </body>
</body></worldbody></mujoco>
"""


class TestPostureBallJacobian(absltest.TestCase):
    """Independent manifold differences exercise the error and QP objective."""

    def _setup_task(self, xml, target_angle=0.7, current_angle=1.1):
        model = mujoco.MjModel.from_xml_string(xml)
        configuration = Configuration(model)
        ball = model.joint("ball").id
        qadr = model.jnt_qposadr[ball]
        target = model.qpos0.copy()
        current = model.qpos0.copy()
        mujoco.mju_axisAngle2Quat(
            target[qadr : qadr + 4], np.array([1.0, 0.0, 0.0]), target_angle
        )
        mujoco.mju_axisAngle2Quat(
            current[qadr : qadr + 4], np.array([0.0, 1.0, 0.0]), current_angle
        )
        configuration.update(current)
        task = PostureTask(model, cost=np.linspace(0.5, 2.0, model.nv), gain=0.8)
        task.set_target(target)
        return model, configuration, task

    def _difference(self, model, configuration, function):
        nominal = configuration.q.copy()
        result = []
        step = 1e-6
        for index in range(model.nv):
            velocity = np.eye(model.nv)[index]
            positive = nominal.copy()
            negative = nominal.copy()
            mujoco.mj_integratePos(model, positive, velocity, step)
            mujoco.mj_integratePos(model, negative, velocity, -step)
            configuration.update(positive)
            plus = function()
            configuration.update(negative)
            minus = function()
            result.append((plus - minus) / (2 * step))
        configuration.update(nominal)
        return np.asarray(result).T

    def test_ball_block_matches_error_finite_differences(self):
        for target_angle, current_angle in ((0.7, 1.1), (-0.4, 2.0), (1e-4, 2e-4)):
            with self.subTest(target=target_angle, current=current_angle):
                model, configuration, task = self._setup_task(
                    SINGLE_BALL, target_angle, current_angle
                )
                analytic = task.compute_jacobian(configuration)
                expected = self._difference(
                    model, configuration, lambda: task.compute_error(configuration)
                )
                np.testing.assert_allclose(analytic, expected, atol=2e-8, rtol=2e-8)

    def test_mixed_model_preserves_other_blocks_and_ignored_free_joint(self):
        model, configuration, task = self._setup_task(MIXED_JOINTS)
        actual = task.compute_jacobian(configuration)
        expected = self._difference(
            model, configuration, lambda: task.compute_error(configuration)
        )
        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-8)
        np.testing.assert_array_equal(actual[:, :6], np.zeros((model.nv, 6)))
        for name in ("hinge", "slide"):
            index = model.jnt_dofadr[model.joint(name).id]
            np.testing.assert_array_equal(actual[:, index], np.eye(model.nv)[:, index])

    def test_qp_linear_term_matches_anisotropic_cost_gradient(self):
        for xml in (SINGLE_BALL, MIXED_JOINTS):
            with self.subTest(xml=xml):
                model, configuration, task = self._setup_task(xml)
                objective = task.compute_qp_objective(configuration)

                def error_cost():
                    residual = task.cost * task.compute_error(configuration)
                    return 0.5 * task.gain * float(residual @ residual)

                expected = self._difference(model, configuration, error_cost)
                np.testing.assert_allclose(objective.c, expected, atol=2e-8, rtol=2e-8)

    def test_equal_target_retains_identity_projection(self):
        for xml in (SINGLE_BALL, MIXED_JOINTS):
            with self.subTest(xml=xml):
                model, configuration, task = self._setup_task(xml)
                task.set_target_from_configuration(configuration)
                expected = np.eye(model.nv)
                if xml == MIXED_JOINTS:
                    expected[:, :6] = 0
                np.testing.assert_allclose(
                    task.compute_jacobian(configuration), expected, atol=1e-14
                )

    def test_jacobian_without_target_retains_legacy_projection(self):
        model, configuration, _ = self._setup_task(MIXED_JOINTS)
        task = PostureTask(model, cost=1.0)
        expected = np.eye(model.nv)
        expected[:, :6] = 0
        np.testing.assert_array_equal(task.compute_jacobian(configuration), expected)


if __name__ == "__main__":
    absltest.main()
