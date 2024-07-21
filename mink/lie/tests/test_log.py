"""Tests for log6 and Jlog6 operations."""
from typing import Type
from absl.testing import absltest, parameterized
import numpy as np
import pinocchio as pin
from mink import lie
from ..log import log6, Jlog6  # Import your implementations
from ..base import MatrixLieGroup

@parameterized.named_parameters(
    ("SE3", lie.SE3),
)
class TestLogOperations(parameterized.TestCase):
    def test_log6_matches_pinocchio(self, group: Type[MatrixLieGroup]):
        """Check that log6 matches Pinocchio's implementation."""
        for _ in range(100):  # Test with 100 random transformations
            X = group.sample_uniform()
            Y = group.sample_uniform()

            # Compute log6 using your implementation
            R = (X.inverse() @ Y).rotation()
            p = (X.inverse() @ Y).translation()
            your_log6 = log6(R.as_matrix(), p)

            # Compute log6 using Pinocchio
            pin_X = pin.SE3(X.as_matrix())
            pin_Y = pin.SE3(Y.as_matrix())
            pin_log6 = pin.log6(pin_X.inverse() * pin_Y)

            np.testing.assert_allclose(your_log6, pin_log6, atol=1e-6)

    def test_Jlog6_matches_pinocchio(self, group: Type[MatrixLieGroup]):
        """Check that Jlog6 matches Pinocchio's implementation."""
        for _ in range(100):  # Test with 100 random transformations
            X = group.sample_uniform()
            Y = group.sample_uniform()

            # Compute Jlog6 using your implementation
            R = (X.inverse() @ Y).rotation()
            p = (X.inverse() @ Y).translation()
            your_Jlog6 = Jlog6(R.as_matrix(), p)

            # Compute Jlog6 using Pinocchio
            pin_X = pin.SE3(X.as_matrix())
            pin_Y = pin.SE3(Y.as_matrix())
            pin_Jlog6 = pin.Jlog6(pin.SE3((pin_X.inverse() * pin_Y).homogeneous))

            np.testing.assert_allclose(your_Jlog6, pin_Jlog6, atol=1e-6)

    def test_log6_identity(self, group: Type[MatrixLieGroup]):
        """Check that log6 of identity is zero."""
        X = group.identity()
        R = X.rotation()
        p = X.translation()

        your_log6 = log6(R.as_matrix(), p)
        expected_log6 = np.zeros(6)

        np.testing.assert_allclose(your_log6, expected_log6, atol=1e-6)

    def test_Jlog6_small_rotation(self, group: Type[MatrixLieGroup]):
        """Check Jlog6 for small rotations."""
        small_rotation = group.exp(np.array([1e-6, 1e-6, 1e-6, 0, 0, 0]))
        R = small_rotation.rotation()
        p = small_rotation.translation()

        your_Jlog6 = Jlog6(R.as_matrix(), p)
        pin_small_rotation = pin.SE3(small_rotation.as_matrix())
        pin_Jlog6 = pin.Jlog6(pin_small_rotation)

        np.testing.assert_allclose(your_Jlog6, pin_Jlog6, atol=1e-6)

if __name__ == "__main__":
    absltest.main()
