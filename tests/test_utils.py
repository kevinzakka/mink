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
        q = utils.custom_configuration_vector(
            self.model, key_name=None, **custom_joints
        )
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

    def test_get_subtree_joint_ids(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body name="b1" pos=".1 -.1 0">
              <joint type="free" name="b1/free"/>
              <geom name="b1/g1" type="sphere" size=".1" mass=".1"/>
              <body name="b2">
                <joint type="slide" name="b2/sx" axis="1 0 0"/>
                <joint type="slide" name="b2/sy" axis="0 1 0"/>
                <joint type="hinge" name="b2/hz" axis="0 0 1"/>
                <geom name="b2/g1" type="sphere" size=".1" mass=".1"/>
                <body name="b2a">
                  <inertial pos="0 0 0" mass=".1" diaginertia="1 1 1"/>
                  <body name="b2b">
                    <joint type="hinge" name="b2b/hinge"/>
                    <geom name="b2b/g1" type="sphere" size=".1" mass=".1"/>
                  </body>
                </body>
              </body>
            </body>
            <body name="b3" pos="1 1 1">
              <joint type="free" name="b3/free"/>
              <geom name="b3/g1" type="sphere" size=".1" mass=".1"/>
              <body name="b4">
                <joint type="hinge" name="b4/hinge" range="0 1.57" limited="true"/>
                <geom name="b4/g1" type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
            <body name="jointless">
              <inertial pos="0 0 0" mass=".1" diaginertia="1 1 1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)

        # Subtree with a multi-joint body, a jointless intermediate body, and a
        # deeper descendant. Sibling subtree (b3/b4) must be excluded.
        b1_id = model.body("b1").id
        actual = utils.get_subtree_joint_ids(model, b1_id)
        expected = [
            model.joint(n).id
            for n in ["b1/free", "b2/sx", "b2/sy", "b2/hz", "b2b/hinge"]
        ]
        self.assertSetEqual(set(actual), set(expected))
        for n in ["b3/free", "b4/hinge"]:
            self.assertNotIn(model.joint(n).id, actual)

        # Querying a body inside b1 returns only its own subtree.
        b2_id = model.body("b2").id
        actual = utils.get_subtree_joint_ids(model, b2_id)
        expected = [model.joint(n).id for n in ["b2/sx", "b2/sy", "b2/hz", "b2b/hinge"]]
        self.assertSetEqual(set(actual), set(expected))
        self.assertNotIn(model.joint("b1/free").id, actual)

        # Sibling subtree.
        b3_id = model.body("b3").id
        actual = utils.get_subtree_joint_ids(model, b3_id)
        expected = [model.joint(n).id for n in ["b3/free", "b4/hinge"]]
        self.assertSetEqual(set(actual), set(expected))

        # Body with no joint anywhere in its subtree.
        jointless_id = model.body("jointless").id
        self.assertListEqual(utils.get_subtree_joint_ids(model, jointless_id), [])

        # World root returns every joint.
        world_id = 0
        actual = utils.get_subtree_joint_ids(model, world_id)
        self.assertSetEqual(set(actual), set(range(model.njnt)))

    def test_get_joint_actuator_id(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body name="root">
              <joint type="free" name="root/free"/>
              <geom type="sphere" size=".05" mass=".1"/>
              <body name="arm">
                <joint type="hinge" name="arm/hinge"/>
                <joint type="slide" name="arm/slide"/>
                <geom type="sphere" size=".05" mass=".1"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <motor name="arm/hinge" joint="arm/hinge"/>
            <position name="root/free" joint="root/free" kp="10"/>
          </actuator>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)

        hinge_id = model.joint("arm/hinge").id
        free_id = model.joint("root/free").id
        slide_id = model.joint("arm/slide").id

        self.assertEqual(
            utils.get_joint_actuator_id(model, hinge_id), model.actuator("arm/hinge").id
        )
        self.assertEqual(
            utils.get_joint_actuator_id(model, free_id), model.actuator("root/free").id
        )
        self.assertIsNone(utils.get_joint_actuator_id(model, slide_id))

    def test_get_subtree_actuator_ids(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body name="b1">
              <joint type="free" name="b1/free"/>
              <geom type="sphere" size=".05" mass=".1"/>
              <body name="b2">
                <joint type="slide" name="b2/sx" axis="1 0 0"/>
                <joint type="hinge" name="b2/hz" axis="0 0 1"/>
                <geom type="sphere" size=".05" mass=".1"/>
                <body name="b2a">
                  <joint type="hinge" name="b2a/hinge"/>
                  <geom type="sphere" size=".05" mass=".1"/>
                </body>
              </body>
            </body>
            <body name="b3">
              <joint type="free" name="b3/free"/>
              <geom type="sphere" size=".05" mass=".1"/>
              <body name="b4">
                <joint type="hinge" name="b4/hinge"/>
                <geom type="sphere" size=".05" mass=".1"/>
              </body>
            </body>
            <body name="actuatorless">
              <joint type="hinge" name="actuatorless/hinge"/>
              <geom type="sphere" size=".05" mass=".1"/>
            </body>
          </worldbody>
          <actuator>
            <motor name="b1/free" joint="b1/free"/>
            <motor name="b2/hz" joint="b2/hz"/>
            <motor name="b2a/hinge" joint="b2a/hinge"/>
            <motor name="b3/free" joint="b3/free"/>
            <motor name="b4/hinge" joint="b4/hinge"/>
          </actuator>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)

        b1_id = model.body("b1").id
        actual = utils.get_subtree_actuator_ids(model, b1_id)
        expected = [
            model.actuator("b1/free").id,
            model.actuator("b2/hz").id,
            model.actuator("b2a/hinge").id,
        ]
        self.assertSetEqual(set(actual), set(expected))

        b3_id = model.body("b3").id
        actual = utils.get_subtree_actuator_ids(model, b3_id)
        expected = [model.actuator("b3/free").id, model.actuator("b4/hinge").id]
        self.assertSetEqual(set(actual), set(expected))

        actuatorless_id = model.body("actuatorless").id
        self.assertListEqual(utils.get_subtree_actuator_ids(model, actuatorless_id), [])

        world_id = 0
        actual = utils.get_subtree_actuator_ids(model, world_id)
        self.assertSetEqual(set(actual), set(range(model.nu)))


if __name__ == "__main__":
    absltest.main()
