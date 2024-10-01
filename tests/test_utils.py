"""Tests for utils.py."""

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import utils
from mink.exceptions import InvalidKeyframe, InvalidMocapBody


class TestUtils(absltest.TestCase):
    """Test utility functions."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("g1_mj_description")

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        self.q0 = self.data.qpos.copy()

    def test_custom_configuration_vector_throws_error_if_keyframe_invalid(self):
        with self.assertRaises(InvalidKeyframe):
            utils.custom_configuration_vector(self.model, "stand123")

    def test_custom_configuration_vector_from_keyframe(self):
        q = utils.custom_configuration_vector(self.model, "stand")
        np.testing.assert_allclose(q, self.model.key("stand").qpos)

    def test_custom_configuration_vector_raises_error_if_jnt_shape_invalid(self):
        with self.assertRaises(ValueError):
            utils.custom_configuration_vector(
                self.model,
                "stand",
                left_ankle_pitch_joint=(0.1, 0.1),
            )

    def test_custom_configuration_vector(self):
        custom_joints = dict(
            left_ankle_pitch_joint=0.2,  # Hinge.
            right_ankle_roll_joint=0.1,  # Slide.
        )
        q = utils.custom_configuration_vector(self.model, **custom_joints)
        q_expected = self.q0.copy()
        for name, value in custom_joints.items():
            qid = self.model.jnt_qposadr[self.model.joint(name).id]
            q_expected[qid] = value
        np.testing.assert_array_almost_equal(q, q_expected)

    def test_move_mocap_to_frame_throws_error_if_body_not_mocap(self):
        with self.assertRaises(InvalidMocapBody):
            utils.move_mocap_to_frame(
                self.model,
                self.data,
                "left_ankle_roll_link",
                "unused_frame_name",
                "unused_frame_type",
            )

    def test_move_mocap_to_frame(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body pos=".1 -.1 0">
              <joint type="free" name="floating"/>
              <geom type="sphere" size=".1" mass=".1"/>
              <body name="test">
                <joint type="hinge" name="hinge" range="0 1.57" limited="true"/>
                <geom type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
            <body name="mocap" mocap="true" pos=".5 1 5" quat="1 1 0 0">
              <geom type="sphere" size=".1" mass=".1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        body_pos = data.body("test").xpos
        body_quat = np.empty(4)
        mujoco.mju_mat2Quat(body_quat, data.body("test").xmat)

        # Initially not the same.
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(data.body("mocap").xpos, body_pos)
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(data.body("mocap").xquat, body_quat)

        utils.move_mocap_to_frame(model, data, "mocap", "test", "body")
        mujoco.mj_forward(model, data)

        # Should now be the same.
        np.testing.assert_allclose(data.body("mocap").xpos, body_pos)
        np.testing.assert_allclose(data.body("mocap").xquat, body_quat)

    def test_get_freejoint_dims(self):
        q_ids, v_ids = utils.get_freejoint_dims(self.model)
        np.testing.assert_allclose(
            np.asarray(q_ids),
            np.asarray(list(range(0, 7))),
        )
        np.testing.assert_allclose(
            np.asarray(v_ids),
            np.asarray(list(range(0, 6))),
        )

    def test_get_subtree_geom_ids(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body name="b1" pos=".1 -.1 0">
              <joint type="free"/>
              <geom name="b1/g1" type="sphere" size=".1" mass=".1"/>
              <geom name="b1/g2" type="sphere" size=".1" mass=".1" pos="0 0 .5"/>
              <body name="b2">
                <joint type="hinge" range="0 1.57" limited="true"/>
                <geom name="b2/g1" type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
            <body name="b3" pos="1 1 1">
              <joint type="free"/>
              <geom name="b3/g1" type="sphere" size=".1" mass=".1"/>
              <body name="b4">
                <joint type="hinge" range="0 1.57" limited="true"/>
                <geom name="b4/g1" type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
            <body name="geomless">
              <inertial pos="0 0 0" mass=".1" diaginertia="1 1 1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        b1_id = model.body("b1").id
        actual_geom_ids = utils.get_subtree_geom_ids(model, b1_id)
        geom_names = ["b1/g1", "b1/g2", "b2/g1"]
        expected_geom_ids = [model.geom(g).id for g in geom_names]
        self.assertSetEqual(set(actual_geom_ids), set(expected_geom_ids))
        b3_id = model.body("b3").id
        actual_geom_ids = utils.get_subtree_geom_ids(model, b3_id)
        geom_names = ["b3/g1", "b4/g1"]
        expected_geom_ids = [model.geom(g).id for g in geom_names]
        self.assertSetEqual(set(actual_geom_ids), set(expected_geom_ids))
        geomless_id = model.body("geomless").id
        actual_geom_ids = utils.get_subtree_geom_ids(model, geomless_id)
        self.assertListEqual(actual_geom_ids, [])
        world_id = 0
        actual_geom_ids = utils.get_subtree_geom_ids(model, world_id)
        expected_geom_ids = [i for i in range(model.ngeom)]
        self.assertSetEqual(set(actual_geom_ids), set(expected_geom_ids))

    def test_get_subtree_body_ids(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body name="b1" pos=".1 -.1 0">
              <joint type="free"/>
              <geom name="b1/g1" type="sphere" size=".1" mass=".1"/>
              <geom name="b1/g2" type="sphere" size=".1" mass=".1" pos="0 0 .5"/>
              <body name="b3">
                <joint type="hinge" range="0 1.57" limited="true"/>
                <geom name="b3/g1" type="sphere" size=".1" mass=".1"/>
                <body name="b4" pos="1 1 1">
                    <geom name="b4/g1" type="sphere" size=".1" mass=".1"/>
                </body>
              </body>
              <body name="b2" pos="1 1 1">
                <geom name="b2/g1" type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
            <body name="b5" pos="1 1 1">
              <joint type="free"/>
              <geom name="b5/g1" type="sphere" size=".1" mass=".1"/>
              <body name="b6">
                <joint type="hinge" range="0 1.57" limited="true"/>
                <geom name="b6/g1" type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        b1_id = model.body("b1").id
        actual_body_ids = utils.get_subtree_body_ids(model, b1_id)
        body_names = ["b1", "b3", "b4", "b2"]
        expected_body_ids = [model.body(b).id for b in body_names]
        self.assertSetEqual(set(actual_body_ids), set(expected_body_ids))
        b5_id = model.body("b5").id
        actual_body_ids = utils.get_subtree_body_ids(model, b5_id)
        body_names = ["b5", "b6"]
        expected_body_ids = [model.body(b).id for b in body_names]
        self.assertSetEqual(set(actual_body_ids), set(expected_body_ids))
        world_id = 0
        actual_body_ids = utils.get_subtree_body_ids(model, world_id)
        expected_body_ids = [i for i in range(model.nbody)]
        self.assertSetEqual(set(actual_body_ids), set(expected_body_ids))


if __name__ == "__main__":
    absltest.main()
