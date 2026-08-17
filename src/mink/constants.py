import mujoco

SUPPORTED_FRAMES = ("body", "geom", "site")

FRAME_TO_ENUM = {
    "body": mujoco.mjtObj.mjOBJ_BODY,
    "geom": mujoco.mjtObj.mjOBJ_GEOM,
    "site": mujoco.mjtObj.mjOBJ_SITE,
}
FRAME_TO_JAC_FUNC = {
    "body": mujoco.mj_jacBody,
    "geom": mujoco.mj_jacGeom,
    "site": mujoco.mj_jacSite,
}
FRAME_TO_POS_ATTR = {
    "body": "xpos",
    "geom": "geom_xpos",
    "site": "site_xpos",
}
FRAME_TO_XMAT_ATTR = {
    "body": "xmat",
    "geom": "geom_xmat",
    "site": "site_xmat",
}
FRAME_TO_COUNT_ATTR = {
    "body": "nbody",
    "geom": "ngeom",
    "site": "nsite",
}


def dof_width(joint_type: int) -> int:
    """Get the dimensionality of the joint in qvel."""
    return {
        mujoco.mjtJoint.mjJNT_FREE.value: 6,
        mujoco.mjtJoint.mjJNT_BALL.value: 3,
        mujoco.mjtJoint.mjJNT_SLIDE.value: 1,
        mujoco.mjtJoint.mjJNT_HINGE.value: 1,
    }[joint_type]


def qpos_width(joint_type: int) -> int:
    """Get the dimensionality of the joint in qpos."""
    return {
        mujoco.mjtJoint.mjJNT_FREE.value: 7,
        mujoco.mjtJoint.mjJNT_BALL.value: 4,
        mujoco.mjtJoint.mjJNT_SLIDE.value: 1,
        mujoco.mjtJoint.mjJNT_HINGE.value: 1,
    }[joint_type]


def constraint_width(constraint: int) -> int:
    """Get the dimensionality of an equality constraint in the efc* arrays."""
    return {
        mujoco.mjtEq.mjEQ_CONNECT.value: 3,
        mujoco.mjtEq.mjEQ_WELD.value: 6,
        mujoco.mjtEq.mjEQ_JOINT.value: 1,
        mujoco.mjtEq.mjEQ_TENDON.value: 1,
    }[constraint]
