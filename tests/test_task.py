"""Tests for task.py."""

import numpy as np
from absl.testing import absltest

from mink.exceptions import InvalidDamping, InvalidGain, InvalidPenalty
from mink.tasks.task import DEFAULT_ELASTIC_PENALTY, Objective, Task


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

    def test_task_is_not_elastic_by_default(self):
        task = Task(cost=np.ones(3))  # type: ignore
        self.assertFalse(task.elastic)
        self.assertEqual(task.penalty, DEFAULT_ELASTIC_PENALTY)

    def test_task_throws_error_if_penalty_invalid(self):
        for penalty in (0.0, -1.0, np.nan, np.inf, [1.0, 0.0, 1.0], [1.0, np.nan, 1.0]):
            with self.subTest(penalty=penalty):
                with self.assertRaises(InvalidPenalty):
                    Task(cost=np.ones(3), elastic=True, penalty=penalty)  # type: ignore

    def test_task_throws_error_if_penalty_shape_invalid(self):
        for penalty in ([1.0, 1.0], np.ones((3, 1)), np.ones((1, 3))):
            with self.subTest(shape=np.shape(penalty)):
                with self.assertRaises(InvalidPenalty):
                    Task(cost=np.ones(3), elastic=True, penalty=penalty)  # type: ignore

    def test_task_throws_error_if_elastic_with_lm_damping(self):
        with self.assertRaises(InvalidDamping):
            Task(cost=np.ones(3), lm_damping=1.0, elastic=True)  # type: ignore

    def test_scalar_penalty_is_broadcast(self):
        for penalty in (5.0, [5.0], np.array(5.0)):
            with self.subTest(penalty=penalty):
                task = Task(cost=np.ones(3), elastic=True, penalty=penalty)  # type: ignore
                self.assertIsInstance(task.penalty, float)
                np.testing.assert_array_equal(task._penalty_vector(3), np.full(3, 5.0))

    def test_per_component_penalty_is_kept(self):
        penalty = np.array([1.0, 2.0, 3.0])
        task = Task(cost=np.ones(3), elastic=True, penalty=penalty)  # type: ignore
        penalty[0] = 100.0  # The task holds its own copy.
        np.testing.assert_array_equal(task._penalty_vector(3), [1.0, 2.0, 3.0])

    def test_penalty_vector_throws_error_on_dimension_mismatch(self):
        task = Task(cost=np.ones(3), elastic=True, penalty=[1.0, 2.0, 3.0])  # type: ignore
        with self.assertRaises(InvalidPenalty):
            task._penalty_vector(4)

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
