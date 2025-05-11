"""Tests for solve_ik.py."""

import numpy as np
from absl.testing import absltest
from numpy.linalg import norm
from robot_descriptions.loaders.mujoco import load_robot_description

import mink
import mujoco

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

    def test_ignores_configuration_limits(self):
        """IK ignores configuration limits if flag is set."""
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

    def test_solve_ik_with_reduced_configuration(self):
        """Test that solve_ik works with ReducedConfiguration having fewer DoFs."""

        # Use just 2 joints for a reduced configuration
        relevant_joints = ["shoulder_pan_joint", "elbow_joint"]
        qpos_indices = np.array([self.model.jnt_qposadr[self.model.joint(j).id] for j in relevant_joints])
        qvel_indices = np.array([self.model.jnt_dofadr[self.model.joint(j).id] for j in relevant_joints])

        class ReducedConfiguration(mink.Configuration):
            def __init__(self, model):
                super().__init__(model)
                self._qpos_indices = qpos_indices
                self._qvel_indices = qvel_indices

            @property
            def q(self):
                return self.data.qpos[self._qpos_indices].copy()

            @q.setter
            def q(self, value):
                self.data.qpos[self._qpos_indices] = value

            @property
            def dq(self):
                return self.data.qvel[self._qvel_indices].copy()

            @dq.setter
            def dq(self, value):
                self.data.qvel[self._qvel_indices] = value

            def get_frame_jacobian(self, frame_name, frame_type):
                full_jacobian = super().get_frame_jacobian(frame_name, frame_type)
                return full_jacobian[:, self._qvel_indices]

            def integrate_inplace(self, velocity, dt):
                full_velocity = np.zeros(self.model.nv)
                full_velocity[self._qvel_indices] = velocity
                super().integrate_inplace(full_velocity, dt)


            def check_limits(self, tol: float = 1e-6, safety_break: bool = True) -> None:
                """Check that the current configuration is within bounds for relevant joints."""
                for idx, jnt in enumerate(self.relevant_joints):
                    jnt_type = self.model.jnt_type[jnt]
                    if jnt_type == mujoco.mjtJoint.mjJNT_FREE or not self.model.jnt_limited[jnt]:
                        continue
                    qval = self.q[idx]  # index into reduced q
                    qmin = self.model.jnt_range[jnt, 0]
                    qmax = self.model.jnt_range[jnt, 1]
                    if qval < qmin - tol or qval > qmax + tol:
                        if safety_break:
                            raise NotWithinConfigurationLimits(
                                joint_id=jnt,
                                value=qval,
                                lower=qmin,
                                upper=qmax,
                                model=self.model,
                            )
                        else:
                            print(
                                f"Value {qval:.2f} at index {idx} is outside of its limits: "
                                f"[{qmin:.2f}, {qmax:.2f}]"
                            )

            @property
            def nv(self):
                return len(self._qvel_indices)
            

            @property
            def relevant_joints(self) -> np.ndarray:
                """Return joint IDs for the reduced qpos indices."""
                joint_ids = []
                for qpos_idx in self._qpos_indices:
                    jnt = np.where(self.model.jnt_qposadr == qpos_idx)[0]
                    if len(jnt) == 0:
                        raise ValueError(f"No joint found for qpos index {qpos_idx}")
                    joint_ids.append(jnt[0])
                return np.array(joint_ids)

        configuration = ReducedConfiguration(self.model)

        # Use a task that's reachable by those 2 joints
        task = mink.FrameTask(
            "attachment_site", "site", position_cost=1.0, orientation_cost=0.0
        )
        init_transform = configuration.get_transform_frame_to_world("attachment_site", "site")
        offset = mink.SE3.from_translation(np.array([0.02, 0.0, 0.0]))  # small displacement
        task.set_target(init_transform @ offset)

        velocity = mink.solve_ik(
            configuration, [task], limits=[], dt=1e-3, solver="daqp"
        )

        # Check velocity shape matches reduced nv
        self.assertEqual(velocity.shape, (configuration.nv,))
        # Check it's not all-zero (since target != current)
        self.assertFalse(np.allclose(velocity, 0.0))


if __name__ == "__main__":
    absltest.main()
