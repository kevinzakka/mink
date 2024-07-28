import numpy as np


def get_epsilon(dtype: np.dtype) -> float:
    return {
        np.dtype("float32"): 1e-5,
        np.dtype("float64"): 1e-10,
    }[dtype]


def skew(x: np.ndarray) -> np.ndarray:
    assert x.shape == (3,)
    wx, wy, wz = x
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ],
        dtype=x.dtype,
    )
