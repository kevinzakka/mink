"""Tests for relative_frame_task.py."""

import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import SE3, Configuration
from mink.tasks import FrameTask, RelativeFrameTask, TargetNotSet, TaskDefinitionError


class TestRelativeFrameTask(absltest.TestCase):
    """Test consistency of the relative frame task."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("g1_mj_description")

    def setUp(self):
        self.configuration = Configuration(self.model)

        np.random.seed(42)
        self.T_wt = SE3.sample_uniform()

    def test_cost_correctly_broadcast(self):
        task = RelativeFrameTask(
            frame_name="pelvis",
            frame_type="body",
            root_name="torso_link",
            root_type="body",
            position_cost=1.0,
            orientation_cost=5.0,
        )
        np.testing.assert_array_equal(task.cost, np.array([1, 1, 1, 5, 5, 5]))

        task = RelativeFrameTask(
            frame_name="pelvis",
            frame_type="body",
            root_name="torso_link",
            root_type="body",
            position_cost=[1.0, 2.0, 3.0],
            orientation_cost=[5.0, 6.0, 7.0],
        )
        np.testing.assert_array_equal(task.cost, np.array([1, 2, 3, 5, 6, 7]))

    def test_task_raises_error_if_cost_dim_invalid(self):
        with self.assertRaises(TaskDefinitionError):
            RelativeFrameTask(
                frame_name="pelvis",
                frame_type="body",
                root_name="torso_link",
                root_type="body",
                position_cost=[1.0, 2.0],
                orientation_cost=2.0,
            )
        with self.assertRaises(TaskDefinitionError):
            RelativeFrameTask(
                frame_name="pelvis",
                frame_type="body",
                root_name="torso_link",
                root_type="body",
                position_cost=7.0,
                orientation_cost=[2.0, 5.0],
            )

    def test_task_raises_error_if_cost_negative(self):
        with self.assertRaises(TaskDefinitionError):
            RelativeFrameTask(
                frame_name="pelvis",
                frame_type="body",
                root_name="torso_link",
                root_type="body",
                position_cost=1.0,
                orientation_cost=-1.0,
            )
        with self.assertRaises(TaskDefinitionError):
            RelativeFrameTask(
                frame_name="pelvis",
                frame_type="body",
                root_name="torso_link",
                root_type="body",
                position_cost=[-1.0, -1.0, -1.0],
                orientation_cost=[1, 2, 3],
            )

    def test_error_without_target(self):
        task = RelativeFrameTask(
            frame_name="pelvis",
            frame_type="body",
            root_name="torso_link",
            root_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        with self.assertRaises(TargetNotSet):
            task.compute_error(self.configuration)

    def test_jacobian_without_target(self):
        task = RelativeFrameTask(
            frame_name="pelvis",
            frame_type="body",
            root_name="torso_link",
            root_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        with self.assertRaises(TargetNotSet):
            task.compute_jacobian(self.configuration)

    def test_set_target_from_configuration(self):
        task = RelativeFrameTask(
            frame_name="pelvis",
            frame_type="body",
            root_name="torso_link",
            root_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        task.set_target_from_configuration(self.configuration)

        pose = self.configuration.get_transform("pelvis", "body", "torso_link", "body")
        np.testing.assert_array_equal(
            task.transform_target_to_root.translation(), pose.translation()
        )
        np.testing.assert_array_equal(
            task.transform_target_to_root.rotation().wxyz, pose.rotation().wxyz
        )

    def test_matches_frame_task(self):
        relative_task = RelativeFrameTask(
            frame_name="pelvis",
            frame_type="body",
            root_name="world",
            root_type="body",
            position_cost=1.0,
            orientation_cost=5.0,
        )
        relative_task.set_target(self.T_wt)

        frame_task = FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=5.0,
        )
        frame_task.set_target(self.T_wt)

        np.testing.assert_allclose(
            frame_task.compute_error(self.configuration),
            -relative_task.compute_error(self.configuration),
        )
        np.testing.assert_allclose(
            frame_task.compute_jacobian(self.configuration),
            -relative_task.compute_jacobian(self.configuration),
        )


if __name__ == "__main__":
    absltest.main()
