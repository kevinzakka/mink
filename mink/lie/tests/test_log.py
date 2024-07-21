"""Tests for log6 and Jlog6 operations."""

from typing import Type

import numpy as np
import pinocchio as pin
from absl.testing import absltest, parameterized

from mink import lie

from ..base import MatrixLieGroup
from ..log import Jlog3, Jlog6, log3, log6


@parameterized.named_parameters(
    ("SE3", lie.SE3),
)
class TestLog6Operations(parameterized.TestCase):
    def test_log6_matches_pinocchio(self, group: Type[MatrixLieGroup]):
        """Check that log6 matches Pinocchio's implementation."""
        for _ in range(100):  # Test with 100 random transformations
            X = group.sample_uniform()

            # Compute log6 using your implementation
            R = X.rotation()
            p = X.translation()
            your_log6 = log6(R.as_matrix(), p)

            # Compute log6 using Pinocchio
            pin_X = pin.SE3(X.as_matrix())
            pin_log6 = pin.log6(pin_X)

            np.testing.assert_allclose(your_log6, pin_log6, atol=1e-6)

    def test_log6_identity(self, group: Type[MatrixLieGroup]):
        """Check that log6 of identity is zero."""
        X = group.identity()
        R = X.rotation()
        p = X.translation()

        your_log6 = log6(R.as_matrix(), p)
        expected_log6 = np.zeros(6)

        np.testing.assert_allclose(your_log6, expected_log6, atol=1e-6)

    def test_Jlog6_small_translation(self, group: Type[MatrixLieGroup]):
        """Check Jlog6 for small rotations."""
        small_rotation = group.exp(np.array([1e-6, 1e-6, 1e-6, 0, 0, 0]))
        R = small_rotation.rotation()
        p = small_rotation.translation()

        your_Jlog6 = Jlog6(R.as_matrix(), p)
        pin_small_rotation = pin.SE3(small_rotation.as_matrix())
        pin_Jlog6 = pin.Jlog6(pin_small_rotation)

        np.testing.assert_allclose(your_Jlog6, pin_Jlog6, atol=1e-6)

    def test_Jlog6_small_rotation(self, group: Type[MatrixLieGroup]):
        """Check Jlog6 for small rotations."""
        small_rotation = group.exp(np.array([0, 0, 0, 1e-6, 1e-6, 1e-6]))
        R = small_rotation.rotation()
        p = small_rotation.translation()

        your_Jlog6 = Jlog6(R.as_matrix(), p)
        pin_small_rotation = pin.SE3(small_rotation.as_matrix())
        pin_Jlog6 = pin.Jlog6(pin_small_rotation)

        np.testing.assert_allclose(your_Jlog6, pin_Jlog6, atol=1e-6)


@parameterized.named_parameters(
    ("SO3", lie.SO3),
)
class TestLog3Operations(parameterized.TestCase):
    def test_log3_matches_pinocchio(self, group: Type[MatrixLieGroup]):
        """Check that log3 matches Pinocchio's implementation."""
        for _ in range(100):  # Test with 100 random rotations
            R = group.sample_uniform()

            # Compute log3 using your implementation
            your_log3 = log3(R.as_matrix())[0]

            # Compute log3 using Pinocchio
            pin_R = R.as_matrix()
            pin_log3 = pin.log3(pin_R)

            np.testing.assert_allclose(your_log3, pin_log3, atol=1e-6)

    def test_Jlog3_matches_pinocchio(self, group: Type[MatrixLieGroup]):
        """Check that Jlog3 matches Pinocchio's implementation."""
        for _ in range(100):  # Test with 100 random rotations
            R = group.sample_uniform()

            # Compute Jlog3 using your implementation
            your_log3 = log3(R.as_matrix())[0]
            your_Jlog3 = Jlog3(np.linalg.norm(your_log3), your_log3)

            # Compute Jlog3 using Pinocchio
            pin_R = R.as_matrix()
            pin_Jlog3 = pin.Jlog3(pin_R)

            np.testing.assert_allclose(your_Jlog3, pin_Jlog3, atol=1e-6)

    def test_log3_identity(self, group: Type[MatrixLieGroup]):
        """Check that log3 of identity is zero."""
        R = group.identity()

        your_log3 = log3(R.as_matrix())[0]
        expected_log3 = np.zeros(3)

        np.testing.assert_allclose(your_log3, expected_log3, atol=1e-6)

    def test_Jlog3_small_rotation(self, group: Type[MatrixLieGroup]):
        """Check Jlog3 for small rotations."""
        small_rotation = group.exp(np.array([1e-3, 1e-3, 1e-3]))

        your_log3 = log3(small_rotation.as_matrix())[0]
        your_Jlog3 = Jlog3(np.linalg.norm(your_log3), your_log3)

        pin_small_rotation = small_rotation.as_matrix()
        pin_Jlog3 = pin.Jlog3(pin_small_rotation)


        np.testing.assert_allclose(your_Jlog3, pin_Jlog3, atol=1e-6)


if __name__ == "__main__":
    absltest.main()
