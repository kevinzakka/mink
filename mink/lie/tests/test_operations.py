"""Tests for general operation definitions."""

from typing import Type

from absl.testing import absltest, parameterized

from mink import lie

from ..base import MatrixLieGroup
from .utils import assert_transforms_close


@parameterized.named_parameters(
    ("SO3", lie.SO3),
    ("SE3", lie.SE3),
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


if __name__ == "__main__":
    absltest.main()
