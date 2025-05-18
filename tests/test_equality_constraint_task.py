"""Tests for equality_constraint_task.py."""

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.exceptions import InvalidConstraint, TaskDefinitionError
from mink.tasks import EqualityConstraintTask
from mink.tasks.equality_constraint_task import _sparse2dense_fallback


class TestEqualityConstraintTask(absltest.TestCase):
    """Test consistency of the equality constraint task."""

    def test_no_equality_constraint_throws(self):
        model = load_robot_description("ur5e_mj_description")
        with self.assertRaises(TaskDefinitionError) as cm:
            EqualityConstraintTask(model=model, cost=1.0)
        expected_error_message = (
            "EqualityConstraintTask no equality constraints found in this model."
        )
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_wrong_cost_dim_throws(self):
        model = load_robot_description("cassie_mj_description")
        # Cassie has 4 equality constraints of type connect. The cost should
        # either be a scalar or a vector of shape (4,).
        with self.assertRaises(TaskDefinitionError) as cm:
            EqualityConstraintTask(model=model, cost=(1, 2))
        expected_error_message = (
            "EqualityConstraintTask cost must be a vector of shape (1,) "
            "or (4,). Got (2,)."
        )
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_cost_correctly_broadcast(self):
        model = load_robot_description("cassie_mj_description")
        # Each connect constraint has dimension 3, so the cost dimension should be 12.
        task = EqualityConstraintTask(model=model, cost=1.0)
        np.testing.assert_array_equal(task.cost, np.full((12,), 1.0))
        task = EqualityConstraintTask(model=model, cost=[1, 2, 3, 4])
        expected_cost = np.asarray([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        np.testing.assert_array_equal(task.cost, expected_cost)

    def test_cost_throws_if_negative(self):
        model = load_robot_description("cassie_mj_description")
        with self.assertRaises(TaskDefinitionError) as cm:
            EqualityConstraintTask(model=model, cost=[-1, 2, 3, 4])
        expected_error_message = "EqualityConstraintTask cost must be >= 0"
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_subset_of_constraints(self):
        model = load_robot_description("cassie_mj_description")
        task = EqualityConstraintTask(
            model=model,
            cost=[23.0, 17.0],
            equalities=[0, 3],
        )
        np.testing.assert_array_equal(
            task.cost,
            np.array([23.0, 23.0, 23.0, 17.0, 17.0, 17.0]),
        )

    def test_duplicate_constraint_ids_throws(self):
        model = load_robot_description("cassie_mj_description")
        with self.assertRaises(TaskDefinitionError) as cm:
            EqualityConstraintTask(model=model, cost=1.0, equalities=[0, 0])
        expected_error_message = "Duplicate equality constraint IDs provided: [0, 0]."
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_inactive_init_constraints_throws_error(self):
        model = load_robot_description("cassie_mj_description")
        model.eq_active0[0] = False
        with self.assertRaises(InvalidConstraint) as cm:
            EqualityConstraintTask(model=model, cost=[1.0, 1.0], equalities=[0, 1])
        expected_error_message = (
            "Equality constraint 0 is not active at initial configuration."
        )
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_subset_of_constraints_with_invalid_name_throws(self):
        model = load_robot_description("cassie_mj_description")
        with self.assertRaises(InvalidConstraint) as cm:
            EqualityConstraintTask(model=model, cost=1.0, equalities=["invalid"])
        expected_error_message = "Equality constraint 'invalid' not found."
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_subset_of_constraints_with_invalid_index_throws(self):
        model = load_robot_description("cassie_mj_description")
        with self.assertRaises(InvalidConstraint) as cm:
            EqualityConstraintTask(model=model, cost=1.0, equalities=[5])
        expected_error_message = (
            "Equality constraint index 5 out of range. Must be in range [0, 4)."
        )
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_zero_error_when_constraint_is_satisfied(self):
        model = load_robot_description("cassie_mj_description")
        task = EqualityConstraintTask(model=model, cost=1.0)
        configuration = Configuration(model)
        configuration.update(model.qpos0)
        error = task.compute_error(configuration)
        np.testing.assert_array_almost_equal(error, np.zeros_like(task.cost), decimal=8)

    def test_zero_cost_same_as_disabling_task(self):
        model = load_robot_description("cassie_mj_description")
        task = EqualityConstraintTask(model=model, cost=0.0)
        configuration = Configuration(model)
        configuration.update_from_keyframe("home")
        objective = task.compute_qp_objective(configuration)
        x = np.random.random(configuration.nv)
        cost = objective.value(x)
        self.assertAlmostEqual(cost, 0.0)

    def test_sparse_jacobian(self):
        model = load_robot_description("cassie_mj_description")
        model.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
        task = EqualityConstraintTask(model=model, cost=1.0)
        configuration = Configuration(model)
        configuration.update_from_keyframe("home")
        jacobian = task.compute_jacobian(configuration)
        self.assertEqual(jacobian.shape, (12, configuration.nv))

    @absltest.skipIf(mujoco.__version__ < "3.2.5", "MuJoCo version is less than 3.2.5")
    def test_sparse_jacobian_fallback(self):
        model = load_robot_description("cassie_mj_description")
        model.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        jac_fallback = np.zeros((data.nefc, model.nv))
        _sparse2dense_fallback(
            res=jac_fallback,
            mat=data.efc_J,
            rownnz=data.efc_J_rownnz,
            rowadr=data.efc_J_rowadr,
            colind=data.efc_J_colind,
        )
        jac_native = np.empty((data.nefc, model.nv))
        mujoco.mju_sparse2dense(
            jac_native,
            data.efc_J,
            data.efc_J_rownnz,
            data.efc_J_rowadr,
            data.efc_J_colind,
        )
        np.testing.assert_array_equal(jac_fallback, jac_native)


if __name__ == "__main__":
    absltest.main()
