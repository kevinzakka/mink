import numpy as np

from mink.lie import SE3, SO3
from mink.lie.base import MatrixLieGroup


def assert_transforms_close(a: MatrixLieGroup, b: MatrixLieGroup) -> None:
    np.testing.assert_allclose(a.as_matrix(), b.as_matrix(), atol=1e-7)

    # Account for quaternion double cover (q = -q).
    pa = a.parameters()
    pb = b.parameters()
    if isinstance(a, SO3):
        pa *= np.sign(pa[0])
        pb *= np.sign(pb[0])
    elif isinstance(a, SE3):
        pa[:4] *= np.sign(pa[0])
        pb[:4] *= np.sign(pb[0])

    np.testing.assert_allclose(pa, pb, atol=1e-7)
