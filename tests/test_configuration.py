import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

import mink


class TestConfiguration(absltest.TestCase):
    """Test task various configuration methods work as intended."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self):
        self.q_ref = self.model.key("home").qpos

    def test_initialize_from_keyframe(self):
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

    def test_site_transform_raises_error_if_site_name_is_invalid(self):
        configuration = mink.Configuration(self.model)
        with self.assertRaises(mink.FrameNotFound):
            configuration.get_transform_frame_to_world("invalid_name", "site")

    def test_site_transform_raises_error_if_site_type_is_invalid(self):
        configuration = mink.Configuration(self.model)
        with self.assertRaises(mink.UnsupportedFrameType):
            configuration.get_transform_frame_to_world("invalid_name", "joint")

    def test_update_raises_error_if_keyframe_is_invalid(self):
        configuration = mink.Configuration(self.model)
        with self.assertRaises(mink.KeyframeNotFound):
            configuration.update_from_keyframe("invalid_keyframe")

    def test_integrate(self):
        configuration = mink.Configuration(self.model, self.q_ref)

        dt = 1e-2
        qvel = np.ones((self.model.nv)) * 0.01
        expected_qpos = self.q_ref + dt * qvel
        actual_qpos = configuration.integrate(qvel, dt)
        np.testing.assert_almost_equal(actual_qpos, expected_qpos)
        # qpos shouldn't be modified.
        np.testing.assert_array_equal(configuration.q, self.q_ref)

        # Inplace integration should change qpos.
        configuration.integrate_inplace(qvel, dt)
        np.testing.assert_almost_equal(configuration.q, expected_qpos)


if __name__ == "__main__":
    absltest.main()
