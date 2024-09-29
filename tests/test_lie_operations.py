"""Tests for general operation definitions."""

from typing import Type

import mujoco
import numpy as np
from absl.testing import absltest, parameterized

from mink.exceptions import InvalidMocapBody
from mink.lie.base import MatrixLieGroup
from mink.lie.se3 import SE3
from mink.lie.so3 import SO3

from .utils import assert_transforms_close


@parameterized.named_parameters(
    ("SO3", SO3),
    ("SE3", SE3),
)
class TestOperations(parameterized.TestCase):
    def test_inverse_bijective(self, group: Type[MatrixLieGroup]):
        """Check inverse of inverse."""
        transform = group.sample_uniform()
        assert_transforms_close(transform, transform.inverse().inverse())

    def test_matrix_bijective(self, group: Type[MatrixLieGroup]):
        """Check that we can convert to and from matrices."""
        transform = group.sample_uniform()
        assert_transforms_close(transform, group.from_matrix(transform.as_matrix()))

    def test_log_exp_bijective(self, group: Type[MatrixLieGroup]):
        """Check 1-to-1 mapping for log <=> exp operations."""
        transform = group.sample_uniform()

        tangent = transform.log()
        self.assertEqual(tangent.shape, (group.tangent_dim,))

        exp_transform = group.exp(tangent)
        assert_transforms_close(transform, exp_transform)
        np.testing.assert_allclose(tangent, exp_transform.log())

    def test_adjoint(self, group: Type[MatrixLieGroup]):
        transform = group.sample_uniform()
        omega = np.random.randn(group.tangent_dim)
        assert_transforms_close(
            transform @ group.exp(omega),
            group.exp(transform.adjoint() @ omega) @ transform,
        )

    def test_rminus(self, group: Type[MatrixLieGroup]):
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        T_c = T_a.inverse() @ T_b
        np.testing.assert_allclose(T_b.rminus(T_a), T_c.log())

    def test_lminus(self, group: Type[MatrixLieGroup]):
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        np.testing.assert_allclose(T_a.lminus(T_b), (T_a @ T_b.inverse()).log())

    def test_rplus(self, group: Type[MatrixLieGroup]):
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        T_c = T_a.inverse() @ T_b
        assert_transforms_close(T_a.rplus(T_c.log()), T_b)

    def test_lplus(self, group: Type[MatrixLieGroup]):
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        T_c = T_a @ T_b.inverse()
        assert_transforms_close(T_b.lplus(T_c.log()), T_a)

    def test_jlog(self, group: Type[MatrixLieGroup]):
        state = group.sample_uniform()
        w = np.random.rand(state.tangent_dim) * 1e-4
        state_pert = state.plus(w).log()
        state_lin = state.log() + state.jlog() @ w
        np.testing.assert_allclose(state_pert, state_lin, atol=1e-7)


class TestGroupSpecificOperations(absltest.TestCase):
    """Group specific tests."""

    # SO3.

    def test_so3_rpy_bijective(self):
        T = SO3.sample_uniform()
        assert_transforms_close(T, SO3.from_rpy_radians(*T.as_rpy_radians()))

    def test_so3_raises_error_if_invalid_shape(self):
        with self.assertRaises(ValueError):
            SO3(wxyz=np.random.rand(2))

    def test_so3_copy(self):
        T = SO3.sample_uniform()
        T_c = T.copy()
        np.testing.assert_allclose(T_c.wxyz, T.wxyz)
        T.wxyz[0] = 1.0
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(T_c.wxyz, T.wxyz)

    def test_se3_apply(self):
        T = SE3.sample_uniform()
        v = np.random.rand(3)
        np.testing.assert_allclose(
            T.apply(v), T.as_matrix()[:3, :3] @ v + T.translation()
        )

    # SE3.

    def test_se3_from_mocap_id(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body name="mocap" mocap="true" pos=".5 1 5" quat="1 1 0 0">
              <geom type="sphere" size=".1" mass=".1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        mid = model.body("mocap").mocapid[0]
        pose = SE3.from_mocap_id(data, mocap_id=mid)
        np.testing.assert_allclose(pose.translation(), data.mocap_pos[mid])
        np.testing.assert_allclose(pose.rotation().wxyz, data.mocap_quat[mid])

    def test_se3_from_mocap_name(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body name="mocap" mocap="true" pos=".5 1 5" quat="1 1 0 0">
              <geom type="sphere" size=".1" mass=".1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        pose = SE3.from_mocap_name(model, data, "mocap")
        mid = model.body("mocap").mocapid[0]
        np.testing.assert_allclose(pose.translation(), data.mocap_pos[mid])
        np.testing.assert_allclose(pose.rotation().wxyz, data.mocap_quat[mid])

    def test_se3_from_mocap_name_raises_error_if_body_not_mocap(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body name="test" pos=".5 1 5" quat="1 1 0 0">
              <geom type="sphere" size=".1" mass=".1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        with self.assertRaises(InvalidMocapBody):
            SE3.from_mocap_name(model, data, "test")


if __name__ == "__main__":
    absltest.main()
