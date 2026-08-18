"""Tests for task.py."""

import numpy as np
from absl.testing import absltest

from mink.exceptions import InvalidDamping, InvalidGain
from mink.tasks.task import Objective, Task


class TestTask(absltest.TestCase):
    """Test abstract base class for tasks."""

    def setUp(self):
        """Prepare test fixture."""
        Task.__abstractmethods__ = frozenset()

    def test_task_throws_error_if_gain_negative(self):
        with self.assertRaises(InvalidGain):
            Task(cost=np.zeros(1), gain=-0.5)  # type: ignore

    def test_task_throws_error_if_lm_damping_negative(self):
        with self.assertRaises(InvalidDamping):
            Task(cost=np.zeros(1), gain=1.0, lm_damping=-1.0)  # type: ignore

    def test_objective_value_matches_quadratic_form(self):
        objective = Objective(
            H=np.array([[2.0, 0.5], [0.5, 4.0]]),
            c=np.array([-1.0, 3.0]),
        )
        x = np.array([2.0, -1.0])
        self.assertEqual(
            objective.value(x), 0.5 * x @ objective.H @ x + objective.c @ x
        )


if __name__ == "__main__":
    absltest.main()
