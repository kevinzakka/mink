"""Tests for configuration.py."""

import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description
import mujoco
from typing import Optional

import mink


class TestConfiguration(absltest.TestCase):
    """Test task various configuration methods work as intended."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self):
        self.q_ref = self.model.key("home").qpos

    def test_nq_nv(self):
        configuration = mink.Configuration(self.model)
        self.assertEqual(configuration.nq, self.model.nq)
        self.assertEqual(configuration.nv, self.model.nv)

    def test_initialize_from_q(self):
        configuration = mink.Configuration(self.model, self.q_ref)
        np.testing.assert_array_equal(configuration.q, self.q_ref)

    def test_initialize_from_keyframe(self):
        """Test that keyframe initialization correctly updates the configuration."""
        configuration = mink.Configuration(self.model)
        np.testing.assert_array_equal(configuration.q, np.zeros(self.model.nq))
        configuration.update_from_keyframe("home")
        np.testing.assert_array_equal(configuration.q, self.q_ref)

    def test_site_transform_world_frame(self):
        site_name = "attachment_site"
        configuration = mink.Configuration(self.model)

        # Randomly sample a joint configuration.
        np.random.seed(12345)
        configuration.data.qpos = np.random.uniform(*configuration.model.jnt_range.T)
        configuration.update()

        world_T_site = configuration.get_transform_frame_to_world(site_name, "site")

        expected_translation = configuration.data.site(site_name).xpos
        np.testing.assert_array_equal(world_T_site.translation(), expected_translation)

        expected_xmat = configuration.data.site(site_name).xmat.reshape(3, 3)
        np.testing.assert_almost_equal(
            world_T_site.rotation().as_matrix(), expected_xmat
        )

    def test_site_transform_raises_error_if_frame_name_is_invalid(self):
        """Raise an error when the requested frame does not exist."""
        configuration = mink.Configuration(self.model)
        with self.assertRaises(mink.InvalidFrame):
            configuration.get_transform_frame_to_world("invalid_name", "site")

    def test_site_transform_raises_error_if_frame_type_is_invalid(self):
        """Raise an error when the requested frame type is invalid."""
        configuration = mink.Configuration(self.model)
        with self.assertRaises(mink.UnsupportedFrame):
            configuration.get_transform_frame_to_world("name_does_not_matter", "joint")

    def test_site_jacobian_raises_error_if_frame_name_is_invalid(self):
        """Raise an error when the requested frame does not exist."""
        configuration = mink.Configuration(self.model)
        with self.assertRaises(mink.InvalidFrame):
            configuration.get_frame_jacobian("invalid_name", "site")

    def test_site_jacobian_raises_error_if_frame_type_is_invalid(self):
        """Raise an error when the requested frame type is invalid."""
        configuration = mink.Configuration(self.model)
        with self.assertRaises(mink.UnsupportedFrame):
            configuration.get_frame_jacobian("name_does_not_matter", "joint")

    def test_update_raises_error_if_keyframe_is_invalid(self):
        """Raise an error when the request keyframe does not exist."""
        configuration = mink.Configuration(self.model)
        with self.assertRaises(mink.InvalidKeyframe):
            configuration.update_from_keyframe("invalid_keyframe")

    def test_inplace_integration(self):
        configuration = mink.Configuration(self.model, self.q_ref)

        dt = 1e-3
        qvel = np.ones((self.model.nv))
        # We can use this formula because the ur5e only has hinge joints.
        expected_qpos = self.q_ref + dt * qvel

        # Regular integration should not modify the underlying q.
        qpos = configuration.integrate(qvel, dt)
        np.testing.assert_almost_equal(qpos, expected_qpos)
        np.testing.assert_equal(configuration.q, self.q_ref)

        # Inplace integration should change qpos.
        configuration.integrate_inplace(qvel, dt)
        np.testing.assert_almost_equal(configuration.q, expected_qpos)

    def test_check_limits(self):
        """Check that an error is raised iff a joint limit is exceeded."""
        configuration = mink.Configuration(self.model, q=self.q_ref)
        configuration.check_limits()
        self.q_ref[0] += 1e4  # Move configuration out of bounds.
        configuration.update(q=self.q_ref)
        with self.assertRaises(mink.NotWithinConfigurationLimits):
            configuration.check_limits()
        configuration.check_limits(safety_break=False)  # Should not raise.

    def test_check_limits_freejoint(self):
        model = load_robot_description("g1_mj_description")
        configuration = mink.Configuration(model)
        q = configuration.q.copy()
        q[0] = 1e4  # x-coordinate of freejoint.
        configuration.update(q=q)
        configuration.check_limits(safety_break=True)  # Should not raise.

    def test_reduced_configuration_with_tasks_and_ik(self):
        """Test that ReducedConfiguration works correctly with tasks and IK solver."""
        # Create our own ReducedConfiguration class based on the provided implementation
        class ReducedConfiguration(mink.Configuration):
            def __init__(self, model, data, relevant_qpos_indices, relevant_qvel_indices):
                super().__init__(model=model, q=None)
                self.relevant_qpos_indices = relevant_qpos_indices
                self.relevant_qvel_indices = relevant_qvel_indices

            @property
            def q(self) -> np.ndarray:
                """Return the relevant qpos entries."""
                return self.data.qpos[self.relevant_qpos_indices].copy()

            @q.setter
            def q(self, value: np.ndarray):
                self.data.qpos[self.relevant_qpos_indices] = value

            @property
            def dq(self) -> np.ndarray:
                """Return the relevant qvel entries."""
                return self.data.qvel[self.relevant_qvel_indices].copy()

            @dq.setter
            def dq(self, value: np.ndarray):
                self.data.qvel[self.relevant_qvel_indices] = value

            def update(self, q: Optional[np.ndarray] = None) -> None:
                if q is not None:
                    self.q = q  
                super().update()

            def get_frame_jacobian(self, frame_name: str, frame_type: str) -> np.ndarray:
                full_jacobian = super().get_frame_jacobian(frame_name, frame_type)
                reduced_jacobian = full_jacobian[:, self.relevant_qvel_indices]
                return reduced_jacobian

            def integrate_inplace(self, velocity: np.ndarray, dt: float) -> None:
                full_velocity = np.zeros(self.model.nv)
                full_velocity[self.relevant_qvel_indices] = velocity
                super().integrate_inplace(full_velocity, dt)

            @property
            def nv(self) -> int:
                return len(self.relevant_qvel_indices)

            def check_limits(self, tol: float = 1e-6, safety_break: bool = True) -> None:
                """Check that the current configuration is within bounds for relevant joints."""
                for idx, jnt in enumerate(self.relevant_joints):
                    jnt_type = self.model.jnt_type[jnt]
                    if jnt_type == mujoco.mjtJoint.mjJNT_FREE or not self.model.jnt_limited[jnt]:
                        continue
                    qval = self.q[idx]
                    qmin = self.model.jnt_range[jnt, 0]
                    qmax = self.model.jnt_range[jnt, 1]
                    if qval < qmin - tol or qval > qmax + tol:
                        if safety_break:
                            raise mink.NotWithinConfigurationLimits(
                                joint_id=jnt,
                                value=qval,
                                lower=qmin,
                                upper=qmax,
                                model=self.model,
                            )

            @property
            def relevant_joints(self) -> np.ndarray:
                """Get the joint IDs corresponding to the relevant qpos indices."""
                joint_ids = []
                for qpos_idx in self.relevant_qpos_indices:
                    jnt = np.where(self.model.jnt_qposadr == qpos_idx)[0][0]
                    joint_ids.append(jnt)
                return np.array(joint_ids)


        data = mujoco.MjData(self.model)

        # Create indices for reduced configuration
        relevant_joints = ["shoulder_pan_joint", "elbow_joint"]
        relevant_qpos_indices = np.array([
            self.model.jnt_qposadr[self.model.joint(j).id] 
            for j in relevant_joints
        ])
        relevant_qvel_indices = np.array([
            self.model.jnt_dofadr[self.model.joint(j).id] 
            for j in relevant_joints
        ])

        # Create the reduced configuration
        configuration = ReducedConfiguration(
            self.model, 
            data, 
            relevant_qpos_indices, 
            relevant_qvel_indices
        )
        
        # Test 1: Verify basic properties
        self.assertEqual(configuration.nv, len(relevant_qvel_indices))
        self.assertEqual(len(configuration.q), len(relevant_qpos_indices))
        self.assertEqual(len(configuration.dq), len(relevant_qvel_indices))
        
        # Test 2: Verify Jacobian computation
        jacobian = configuration.get_frame_jacobian("attachment_site", "site")
        self.assertEqual(jacobian.shape, (6, configuration.nv))
        
        # Create a task that will use the reduced configuration
        task = mink.FrameTask(
            "attachment_site", 
            "site", 
            position_cost=1.0, 
            orientation_cost=0.0
        )
        
        # Set a target to ensure we have some error
        init_transform = configuration.get_transform_frame_to_world("attachment_site", "site")
        offset = mink.SE3.from_translation(np.array([0.02, 0.0, 0.0]))
        task.set_target(init_transform @ offset)
        
        # Test 3: Verify QP objective computation with reduced configuration
        objective = task.compute_qp_objective(configuration)
        self.assertEqual(objective.H.shape, (configuration.nv, configuration.nv))
        self.assertEqual(objective.c.shape, (configuration.nv,))
        
        # Test 4: Verify IK solver works with reduced configuration
        velocity = mink.solve_ik(
            configuration, 
            [task], 
            limits=[], 
            dt=1e-3, 
            solver="daqp"
        )
        
        # Verify velocity shape matches reduced configuration
        self.assertEqual(velocity.shape, (configuration.nv,))
        
        # Test 5: Verify the velocity is not all zeros (since target != current)
        self.assertFalse(np.allclose(velocity, 0.0))
        
        # Test 6: Verify integration works with reduced configuration
        dt = 1e-3
        configuration.integrate_inplace(velocity, dt)
        self.assertEqual(configuration.q.shape, (len(relevant_qpos_indices),))
        
        # Test 7: Verify the task error changes after moving
        error_after = task.compute_error(configuration)
        self.assertFalse(np.allclose(error_after, 0.0))
        
        # Test 8: Verify limits checking
        configuration.check_limits()  # Should not raise
        
        # Test 9: Verify update method
        new_q = np.zeros(len(relevant_qpos_indices))
        configuration.update(q=new_q)
        np.testing.assert_array_equal(configuration.q, new_q)


if __name__ == "__main__":
    absltest.main()
