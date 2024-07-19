"""Tests for configuration_limit.py."""

from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration


class TestConfigurationLimit(absltest.TestCase):
    """Test configuration limit."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self):
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("home")


if __name__ == "__main__":
    absltest.main()
