# reference: https://github.com/stack-of-tasks/pinocchio/blob/c989669e255715e2fa2504b3226664bf20de6fb5/include/pinocchio/spatial/explog.hpp

from typing import Tuple

import numpy as np

from .utils import skew


def log3(R: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute the logarithm of a rotation matrix.

    Args:
        R (np.ndarray): 3x3 rotation matrix

    Returns:
        log3 (np.ndarray): 3x1 vector representation of the rotation
        theta (float):
    """
    res = np.zeros(3)

    # Upper and lower bounds for the trace of R
    ub = 3
    lb = -1

    # Compute trace of matrix
    tr = np.trace(R)

    # Determine trace
    trace = np.clip(tr, lb, ub)

    # Determine theta based on given conditions
    angle = np.where(trace <= lb, np.pi, np.arccos((trace - 1) / 2))
    angle = np.where(trace >= ub, 0, angle)

    # Check if theta is larger than pi
    theta_larger_than_pi = angle >= np.pi - 1e-2

    # Create relaxation
    cond_prec = angle > 1e-8
    cphi = -(trace - 1) / 2
    beta = angle * angle / (1 + cphi)
    tmp = (np.diag(R) + cphi) * beta
    t = np.where(cond_prec, angle / np.sin(angle), 1.0) / 2.0

    # Create switching conditions for relaxation
    cond_R1 = 1 if R[2, 1] > R[1, 2] else -1
    cond_R2 = 1 if R[0, 2] > R[2, 0] else -1
    cond_R3 = 1 if R[1, 0] > R[0, 1] else -1

    cond_tmp = np.sqrt(np.maximum(tmp, 0))

    res[0] = np.where(
        theta_larger_than_pi, cond_R1 * cond_tmp[0], t * (R[2, 1] - R[1, 2])
    )
    res[1] = np.where(
        theta_larger_than_pi, cond_R2 * cond_tmp[1], t * (R[0, 2] - R[2, 0])
    )
    res[2] = np.where(
        theta_larger_than_pi, cond_R3 * cond_tmp[2], t * (R[1, 0] - R[0, 1])
    )

    return res, angle


def Jlog3(theta: float, log: np.ndarray) -> np.ndarray:
    """Compute the Jacobian of log3.

    Args:
        theta (float): Angle of rotation
        log (np.ndarray): 3x1 vector representation of the rotation

    Returns:
        np.ndarray: 3x3 Jacobian matrix
    """
    J = np.zeros((3, 3))

    cond_prec = theta < 1e-8
    st = np.sin(theta)
    ct = np.cos(theta)
    st_1mct = st / (1.0 - ct)

    alpha = np.where(
        cond_prec,
        1.0 / 12.0 + theta * theta / 720.0,
        1.0 / (theta * theta) - (st_1mct) / (2 * theta),
    )

    diag = np.where(cond_prec, 0.5 * (2.0 - theta * theta / 6.0), 0.5 * theta * st_1mct)

    # Create result
    J = alpha * np.outer(log, log)
    np.fill_diagonal(J, J.diagonal() + diag)
    J += 0.5 * skew(log)

    return J


def log6(R, p):
    """Compute the logarithm of a SE(3) transformation.

    Args:
        R (np.ndarray): 3x3 rotation matrix
        p (np.ndarray): 3x1 translation vector

    Returns:
        np.ndarray: 6x1 vector representation of the transformation
    """
    res = np.zeros(6)

    # Get rotational component on Lie algebra
    w, theta = log3(R)

    theta2 = theta * theta

    # Create relaxation
    cond_prec = theta < 1e-8

    st = np.sin(theta)
    ct = np.cos(theta)

    alpha = np.where(
        cond_prec, 1 - theta2 / 12 - theta2 * theta2 / 720, theta * st / (2 * (1 - ct))
    )

    beta = np.where(
        cond_prec, 1.0 / 12.0 + theta2 / 720, 1.0 / theta2 - st / (2 * theta * (1 - ct))
    )

    res[:3] = alpha * p - 0.5 * np.cross(w, p) + (beta * np.dot(w, p)) * w
    res[3:] = w
    return res


def Jlog6(R, p):
    """Compute the Jacobian of log6.

    Args:
        R (np.ndarray): 3x3 rotation matrix
        p (np.ndarray): 3x1 translation vector

    Returns:
        np.ndarray: 6x6 Jacobian matrix
    """
    J = np.zeros((6, 6))

    # Compute theta
    w, theta = log3(R)

    # Get blocks of the jacobian
    TL = Jlog3(theta, w)
    BR = TL

    theta2 = theta * theta

    cond_prec = theta < 1e-8
    tinv = 1.0 / theta if not cond_prec else 0
    tinv2 = tinv * tinv
    st = np.sin(theta)
    ct = np.cos(theta)
    inv_2_2ct = 1.0 / (2 * (1.0 - ct)) if not cond_prec else 0

    st_1mct = st / (1.0 - ct) if not cond_prec else 0

    beta = np.where(
        cond_prec, 1.0 / 12.0 + theta2 / 720.0, tinv2 - st * tinv * inv_2_2ct
    )

    beta_dot_over_theta = np.where(
        cond_prec,
        1.0 / 360.0,
        -2.0 * tinv2 * tinv2 + (1.0 + st * tinv) * tinv2 * inv_2_2ct,
    )

    wTp = np.dot(w, p)
    v3_tmp = (beta_dot_over_theta * wTp) * w - (
        theta2 * beta_dot_over_theta + 2.0 * beta
    ) * p

    BL = np.outer(v3_tmp, w)
    BL += beta * np.outer(w, p)
    np.fill_diagonal(BL, BL.diagonal() + wTp * beta)
    BL += skew(0.5 * p)

    TR = np.dot(BL, TL)
    BL = np.zeros((3, 3))

    J[:3, :3] = TL
    J[:3, 3:] = TR
    J[3:, :3] = BL
    J[3:, 3:] = BR
    return J
