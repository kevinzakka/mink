"""Tests for group axioms."""

from typing import Type

from absl.testing import absltest, parameterized

from mink import lie
from mink.lie.base import MatrixLieGroup

from .utils import assert_transforms_close


@parameterized.named_parameters(
    ("SO3", lie.SO3),
    ("SE3", lie.SE3),
)
class TestAxioms(parameterized.TestCase):
    def test_closure(self, group: Type[MatrixLieGroup]):
        transform_a = group.sample_uniform()
        transform_b = group.sample_uniform()
        composed = transform_a @ transform_b
        assert_transforms_close(composed, composed.normalize())
        composed = transform_b @ transform_a
        assert_transforms_close(composed, composed.normalize())
        composed = transform_a @ transform_b
        assert_transforms_close(composed, composed.normalize())

    def test_identity(self, group: Type[MatrixLieGroup]):
        transform = group.sample_uniform()
        identity = group.identity()
        assert_transforms_close(transform, identity @ transform)
        assert_transforms_close(transform, transform @ identity)

    def test_inverse(self, group: Type[MatrixLieGroup]):
        transform = group.sample_uniform()
        identity = group.identity()
        assert_transforms_close(identity, transform.inverse() @ transform)
        assert_transforms_close(identity, transform @ transform.inverse())

    def test_associative(self, group: Type[MatrixLieGroup]):
        transform_a = group.sample_uniform()
        transform_b = group.sample_uniform()
        transform_c = group.sample_uniform()
        assert_transforms_close(
            (transform_a @ transform_b) @ transform_c,
            transform_a @ (transform_b @ transform_c),
        )


if __name__ == "__main__":
    absltest.main()
