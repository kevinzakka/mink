"""Tests for equality_constraint_task.py."""

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration, solve_ik
from mink.exceptions import InvalidConstraint, TaskDefinitionError
from mink.tasks import EqualityConstraintTask


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

    def test_subset_of_constraints_compute_uses_correct_costs(self):
        """Costs are looked up by position in the subset, not by equality id."""
        model = load_robot_description("cassie_mj_description")
        configuration = Configuration(model)
        configuration.update_from_keyframe("home")
        task = EqualityConstraintTask(
            model=model,
            cost=[23.0, 17.0],
            equalities=[0, 3],
        )
        # Previously raised IndexError: cost was indexed by equality id 3.
        error = task.compute_error(configuration)
        self.assertEqual(error.shape, (6,))
        np.testing.assert_array_equal(
            task.cost,
            np.array([23.0, 23.0, 23.0, 17.0, 17.0, 17.0]),
        )
        jacobian = task.compute_jacobian(configuration)
        self.assertEqual(jacobian.shape, (6, model.nv))

    def test_subset_costs_follow_constraint_order(self):
        """Costs map to their equality even when the subset is not in id order."""
        model = load_robot_description("cassie_mj_description")
        configuration = Configuration(model)
        configuration.update_from_keyframe("home")
        task = EqualityConstraintTask(
            model=model,
            cost=[17.0, 23.0],
            equalities=[3, 0],
        )
        # Rows are ordered by equality id: id 0 (cost 23) then id 3 (cost 17), and
        # the ordering is identical before and after the first compute.
        expected = np.array([23.0, 23.0, 23.0, 17.0, 17.0, 17.0])
        np.testing.assert_array_equal(task.cost, expected)
        task.compute_error(configuration)
        np.testing.assert_array_equal(task.cost, expected)

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

    def test_tendon_ik_converges_to_coupled_lengths(self):
        # Fixed tendons with coef 1 have length q, and both reference lengths
        # are zero at qpos0, so the equality reduces to q1 = q2.
        xml = """
        <mujoco>
          <worldbody>
            <body>
              <joint name="j1" type="slide" axis="1 0 0"/>
              <geom type="sphere" size="0.05" mass="0.1"/>
            </body>
            <body pos="0 1 0">
              <joint name="j2" type="slide" axis="1 0 0"/>
              <geom type="sphere" size="0.05" mass="0.1"/>
            </body>
          </worldbody>
          <tendon>
            <fixed name="t1"><joint joint="j1" coef="1"/></fixed>
            <fixed name="t2"><joint joint="j2" coef="1"/></fixed>
          </tendon>
          <equality>
            <tendon name="coupling" tendon1="t1" tendon2="t2"/>
          </equality>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        configuration = Configuration(model, np.array([0.5, -0.3]))
        task = EqualityConstraintTask(model=model, cost=1.0)

        # If tendon kinematics were stale, both lengths would read zero and the
        # final check would pass trivially. Verify the violation is seen.
        error = task.compute_error(configuration)
        self.assertEqual(error.shape, (1,))
        self.assertAlmostEqual(abs(float(error[0])), 0.8)

        # J^T J is rank one (motion along [1, 1] is free); damping selects the
        # minimum-norm correction.
        dt = 0.1
        for _ in range(100):
            velocity = solve_ik(
                configuration, [task], dt=dt, solver="daqp", damping=1e-8
            )
            configuration.integrate_inplace(velocity, dt)

        ten_length = configuration.data.ten_length
        self.assertAlmostEqual(float(ten_length[0]), float(ten_length[1]), places=6)
        error = task.compute_error(configuration)
        np.testing.assert_allclose(error, 0.0, atol=1e-6)


if __name__ == "__main__":
    absltest.main()
