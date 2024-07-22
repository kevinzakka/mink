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


def dof_width(joint_type: mujoco.mjtJoint) -> int:
    """Get the dimensionality of the joint in qvel."""
    return {0: 6, 1: 3, 2: 1, 3: 1}[joint_type]


def qpos_width(joint_type: mujoco.mjtJoint) -> int:
    """Get the dimensionality of the joint in qpos."""
    return {0: 7, 1: 4, 2: 1, 3: 1}[joint_type]
