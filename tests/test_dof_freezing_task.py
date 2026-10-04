"""Tests for dof_freezing_task.py."""

import mujoco_menagerie as mm
import numpy as np
from absl.testing import absltest

from mink import Configuration
from mink.exceptions import TaskDefinitionError
from mink.tasks import DofFreezingTask


class TestDofFreezingTask(absltest.TestCase):
    """Test consistency of the DOF freezing task."""

    @classmethod
    def setUpClass(cls):
        cls.model = mm.load("franka_emika_panda", "panda")

    def setUp(self):
        self.configuration = Configuration(self.model)

    def test_task_raises_error_if_dof_indices_empty(self):
        with self.assertRaises(TaskDefinitionError) as cm:
            DofFreezingTask(model=self.model, dof_indices=[])
        expected_error_message = "DofFreezingTask requires at least one DOF index."
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_task_raises_error_if_dof_index_negative(self):
        with self.assertRaises(TaskDefinitionError) as cm:
            DofFreezingTask(model=self.model, dof_indices=[-1])
        expected_error_message = f"DOF index -1 is out of range [0, {self.model.nv})."
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_task_raises_error_if_dof_index_too_large(self):
        with self.assertRaises(TaskDefinitionError) as cm:
            DofFreezingTask(model=self.model, dof_indices=[self.model.nv])
        expected_error_message = (
            f"DOF index {self.model.nv} is out of range [0, {self.model.nv})."
        )
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_task_raises_error_if_duplicate_dof_indices(self):
        with self.assertRaises(TaskDefinitionError) as cm:
            DofFreezingTask(model=self.model, dof_indices=[0, 1, 0])
        expected_error_message = "Duplicate DOF indices found: [0, 1, 0]."
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_dof_indices_are_sorted(self):
        task = DofFreezingTask(model=self.model, dof_indices=[3, 1, 2])
        np.testing.assert_array_equal(task.dof_indices, [1, 2, 3])

    def test_error_is_always_zero(self):
        task = DofFreezingTask(model=self.model, dof_indices=[0, 1, 2])
        error = task.compute_error(self.configuration)
        np.testing.assert_array_equal(error, np.zeros(3))

    def test_jacobian_shape(self):
        dof_indices = [0, 2, 4]
        task = DofFreezingTask(model=self.model, dof_indices=dof_indices)
        jac = task.compute_jacobian(self.configuration)
        expected_shape = (len(dof_indices), self.model.nv)
        self.assertEqual(jac.shape, expected_shape)

    def test_jacobian_structure_single_dof(self):
        """Test that Jacobian for a single DOF is a single row of the identity."""
        dof_idx = 3
        task = DofFreezingTask(model=self.model, dof_indices=[dof_idx])
        jac = task.compute_jacobian(self.configuration)
        expected_jac = np.zeros((1, self.model.nv))
        expected_jac[0, dof_idx] = 1.0
        np.testing.assert_array_equal(jac, expected_jac)

    def test_jacobian_structure_multiple_dofs(self):
        """Test that Jacobian for multiple DOFs has identity structure."""
        dof_indices = [1, 3, 5]
        task = DofFreezingTask(model=self.model, dof_indices=dof_indices)
        jac = task.compute_jacobian(self.configuration)
        expected_jac = np.zeros((3, self.model.nv))
        for i, dof_idx in enumerate(dof_indices):
            expected_jac[i, dof_idx] = 1.0
        np.testing.assert_array_equal(jac, expected_jac)

    def test_jacobian_unchanged_by_configuration(self):
        """Test that Jacobian doesn't depend on configuration values."""
        dof_indices = [0, 1]
        task = DofFreezingTask(model=self.model, dof_indices=dof_indices)

        # Compute Jacobian at initial configuration.
        jac1 = task.compute_jacobian(self.configuration)

        # Change configuration.
        q_new = self.configuration.q.copy()
        q_new[:3] += np.random.randn(3)
        self.configuration.update(q=q_new)

        # Compute Jacobian at new configuration.
        jac2 = task.compute_jacobian(self.configuration)

        # Jacobians should be identical.
        np.testing.assert_array_equal(jac1, jac2)

    def test_task_dimension_matches_num_dofs(self):
        """Test that task dimension k equals number of frozen DOFs."""
        dof_indices = [0, 2, 4, 6]
        task = DofFreezingTask(model=self.model, dof_indices=dof_indices)
        self.assertEqual(task.k, len(dof_indices))

    def test_cost_dimension_matches_num_dofs(self):
        """Test that cost vector has correct dimension."""
        dof_indices = [1, 3]
        task = DofFreezingTask(model=self.model, dof_indices=dof_indices)
        self.assertEqual(task.cost.shape, (len(dof_indices),))
        # Default cost should be all ones.
        np.testing.assert_array_equal(task.cost, np.ones(len(dof_indices)))

    def test_gain_is_stored(self):
        """Test that custom gain is stored correctly."""
        task = DofFreezingTask(model=self.model, dof_indices=[0, 1], gain=0.5)
        self.assertEqual(task.gain, 0.5)

    def test_all_dofs_can_be_frozen(self):
        """Test that we can freeze all DOFs without error."""
        all_dof_indices = list(range(self.model.nv))
        task = DofFreezingTask(model=self.model, dof_indices=all_dof_indices)
        jac = task.compute_jacobian(self.configuration)
        # Should be the identity matrix.
        np.testing.assert_array_equal(jac, np.eye(self.model.nv))

    def test_qp_objective_with_zero_error(self):
        """Test that QP objective is computed correctly with zero error.

        Since error is always zero, the objective should only depend on the
        Jacobian term: (1/2) * dq^T * (J^T W J) * dq.
        """
        dof_indices = [0, 1]
        task = DofFreezingTask(model=self.model, dof_indices=dof_indices, gain=1.0)
        H, c = task.compute_qp_objective(self.configuration)

        # Get expected values.
        J = task.compute_jacobian(self.configuration)
        W = np.diag(task.cost)
        expected_H = J.T @ W @ J

        # Since error is zero, c should be zero.
        expected_c = np.zeros(self.model.nv)

        np.testing.assert_allclose(H, expected_H)
        np.testing.assert_allclose(c, expected_c)


if __name__ == "__main__":
    absltest.main()
