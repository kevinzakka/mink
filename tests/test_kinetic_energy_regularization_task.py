"""Tests for kinetic_energy_regularization_task.py."""

import numpy as np
import mujoco
from absl.testing import absltest

from mink.exceptions import TaskDefinitionError
from mink.tasks import KineticEnergyRegularizationTask
from mink import Configuration


class TestKineticEnergyRegularizationTask(absltest.TestCase):
    """Test consistency of the kinetic energy regularization task."""

    def test_cost_must_be_nonnegative(self):
        with self.assertRaises(TaskDefinitionError) as cm:
            KineticEnergyRegularizationTask(cost=-1.0)
        self.assertEqual(
            str(cm.exception), "KineticEnergyRegularizationTask cost should be >= 0"
        )

    def test_cost_must_be_scalar(self):
        with self.assertRaises(TaskDefinitionError) as cm:
            KineticEnergyRegularizationTask(cost=[1.0, 2.0])
        self.assertEqual(
            str(cm.exception), "KineticEnergyRegularizationTask cost must be a scalar"
        )

    def test_qp_objective_is_correct(self):
        xml_str = """
        <mujoco model="test">
        </mujoco>
        """

        model = mujoco.MjModel.from_xml_string(xml_str)
        configuration = Configuration(model)
        task = KineticEnergyRegularizationTask(cost=1e-3)
        objective = task.compute_qp_objective(configuration)
        # All dofs are regularized equally, including floating-base coordinates.
        np.testing.assert_array_equal(objective.H, np.eye(configuration.nv) * 1e-3)
        np.testing.assert_array_equal(objective.c, np.zeros(configuration.nv))

    def test_simple_inertia_structure(self):
        xml_str = r"""
<mujoco model="test">
  <worldbody>
    <body name="body1">
      <joint type="slide" axis="1 0 0"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
      <body name="body2">
        <joint type="slide" axis="1 0 0"/>
        <geom type="box" size="0.1 0.1 0.1" mass="2"/>
        <body name="body3">
          <joint type="slide" axis="1 0 0"/>
          <geom type="box" size="0.1 0.1 0.1" mass="3"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        configuration = Configuration(model)
        task = KineticEnergyRegularizationTask(cost=1e-3)
        objective = task.compute_qp_objective(configuration)

        # Theoretical mass matrix for our simple system:
        # - Diagonal terms represent the mass seen by each joint
        # - Off-diagonal terms represent the coupling between joints
        theoretical_mass_matrix = np.array(
            [
                [6.0, 5.0, 3.0],  # Joint 1 affects all masses.
                [5.0, 5.0, 3.0],  # Joint 2 affects masses 2 and 3.
                [3.0, 3.0, 3.0],  # Joint 3 affects only mass 3.
            ]
        )

        # H should be the mass matrix scaled by the cost.
        expected_H = theoretical_mass_matrix * 1e-3
        np.testing.assert_array_equal(objective.H, expected_H)
        np.testing.assert_array_equal(objective.c, np.zeros(3))


if __name__ == "__main__":
    absltest.main()
