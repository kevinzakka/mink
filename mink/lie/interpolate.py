from mink.lie import SE3, SO3


def interpolate(
    p0: SE3 | SO3,
    p1: SE3 | SO3,
    alpha: float = 0.5,
) -> SE3 | SO3:
    assert 0.0 <= alpha <= 1.0
    exp_func = getattr(type(p0), "exp")
    return p0 @ exp_func(alpha * (p0.inverse() @ p1).log())
