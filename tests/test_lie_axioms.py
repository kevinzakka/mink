"""Tests for group axioms."""

from typing import Type

import numpy as np
from absl.testing import absltest, parameterized

from mink import lie
from mink.lie.base import MatrixLieGroup


def assert_transforms_close(a: MatrixLieGroup, b: MatrixLieGroup) -> None:
    np.testing.assert_allclose(a.as_matrix(), b.as_matrix(), atol=1e-7)

    # Account for quaternion double cover (q = -q).
    pa = a.parameters()
    pb = b.parameters()
    if isinstance(a, lie.SO3):
        pa *= np.sign(pa[0])
        pb *= np.sign(pb[0])
    elif isinstance(a, lie.SE3):
        pa[:4] *= np.sign(pa[0])
        pb[:4] *= np.sign(pb[0])

    np.testing.assert_allclose(pa, pb, atol=1e-7)


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
