"""Unit transforms must own their mutable parameter buffers."""

from typing import cast

import numpy as np
from absl.testing import absltest

from mink import SE3, SO3
from mink.lie.base import MatrixLieGroup


class TestIdentityStorage(absltest.TestCase):
    """Mutating a returned identity cannot corrupt independent transforms."""

    def _assert_identity_isolation(self, group, accessor, value):
        first = group.identity()
        original = first.parameters().copy()
        second = group.identity()
        expected_matrix = np.eye(group.matrix_dim)
        try:
            accessor(first)[...] = value
            np.testing.assert_array_equal(second.parameters(), original)
            np.testing.assert_array_equal(group.identity().parameters(), original)
            np.testing.assert_array_equal(second.as_matrix(), expected_matrix)
            np.testing.assert_array_equal(group.identity().as_matrix(), expected_matrix)
        finally:
            # Restore the pristine global buffer after testing the buggy version,
            # so a failure does not change the inputs of other upstream tests.
            first.parameters()[...] = original

    def test_so3_parameters_own_storage(self):
        self._assert_identity_isolation(
            SO3, SO3.parameters, np.array([np.cos(0.4), np.sin(0.4), 0, 0])
        )

    def test_so3_quaternion_field_owns_storage(self):
        self._assert_identity_isolation(
            SO3,
            lambda transform: transform.wxyz,
            np.array([np.cos(0.4), 0, np.sin(0.4), 0]),
        )

    def test_se3_parameters_own_storage(self):
        self._assert_identity_isolation(
            SE3,
            SE3.parameters,
            np.array([np.cos(0.4), 0, 0, np.sin(0.4), 1, -2, 3]),
        )

    def test_se3_translation_view_owns_storage(self):
        self._assert_identity_isolation(SE3, SE3.translation, np.array([1, -2, 3]))

    def test_se3_rotation_view_owns_storage(self):
        self._assert_identity_isolation(
            SE3,
            lambda transform: transform.rotation().parameters(),
            np.array([np.cos(0.4), 0, np.sin(0.4), 0]),
        )

    def test_unit_laws_survive_independent_identity_mutation(self):
        rotation = SO3.from_rpy_radians(0.2, -0.3, 0.4)
        transform = SE3.from_rotation_and_translation(
            rotation, np.array([1.0, -2.0, 3.0])
        )
        for group, element in ((SO3, rotation), (SE3, transform)):
            with self.subTest(group=group.__name__):
                changed_identity = group.identity()
                original = changed_identity.parameters().copy()
                identity = group.identity()
                try:
                    changed_identity.parameters()[:4] = [1, 0, 0, 0]
                    changed_identity.parameters()[1] = np.sin(0.5)
                    changed_identity.parameters()[0] = np.cos(0.5)
                    np.testing.assert_allclose(
                        (
                            cast(MatrixLieGroup, identity)
                            @ cast(MatrixLieGroup, element)
                        ).as_matrix(),
                        element.as_matrix(),
                        rtol=0,
                        atol=1e-14,
                    )
                    np.testing.assert_allclose(
                        (
                            cast(MatrixLieGroup, element)
                            @ cast(MatrixLieGroup, identity)
                        ).as_matrix(),
                        element.as_matrix(),
                        rtol=0,
                        atol=1e-14,
                    )
                finally:
                    changed_identity.parameters()[...] = original


if __name__ == "__main__":
    absltest.main()
