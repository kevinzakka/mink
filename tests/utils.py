import numpy as np

from mink.lie import SE3, SO3
from mink.lie.base import MatrixLieGroup


def assert_transforms_close(a: MatrixLieGroup, b: MatrixLieGroup) -> None:
    np.testing.assert_allclose(a.as_matrix(), b.as_matrix(), atol=1e-7)

    # Account for quaternion double cover (q = -q).
    pa = a.parameters().copy()
    pb = b.parameters().copy()
    if isinstance(a, SO3):
        if np.dot(pa, pb) < 0.0:
            pb = -pb
    elif isinstance(a, SE3):
        if np.dot(pa[:4], pb[:4]) < 0.0:
            pb[:4] = -pb[:4]

    np.testing.assert_allclose(pa, pb, atol=1e-7)
