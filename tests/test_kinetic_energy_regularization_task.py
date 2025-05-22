"""Tests for kinetic_energy_regularization_task.py."""

import mujoco
import numpy as np
from absl.testing import absltest

from mink import Configuration
from mink.exceptions import TaskDefinitionError, IntegrationTimestepNotSet
from mink.tasks import KineticEnergyRegularizationTask


class TestKineticEnergyRegularizationTask(absltest.TestCase):
    """Test consistency of the kinetic energy regularization task."""

    def test_cost_must_be_nonnegative(self):
        with self.assertRaises(TaskDefinitionError) as cm:
            KineticEnergyRegularizationTask(cost=-1.0)
        self.assertEqual(
            str(cm.exception), "KineticEnergyRegularizationTask cost should be >= 0"
        )

    def test_no_dt_set_throws(self):
        task = KineticEnergyRegularizationTask(cost=1.0)
        with self.assertRaises(IntegrationTimestepNotSet) as cm:
            task.compute_qp_objective(Configuration(mujoco.MjModel.from_xml_string("<mujoco/>")))
        self.assertEqual(
            str(cm.exception), "No integration timestep set for KineticEnergyRegularizationTask"
        )

    def test_qp_objective_is_correct(self):
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
        cost = 1e-3
        dt = 0.02
        task = KineticEnergyRegularizationTask(cost=cost)
        task.set_dt(dt)
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

        expected_H = theoretical_mass_matrix * cost / dt**2
        np.testing.assert_array_equal(objective.H, expected_H)
        np.testing.assert_array_equal(objective.c, np.zeros(3))


if __name__ == "__main__":
    absltest.main()
