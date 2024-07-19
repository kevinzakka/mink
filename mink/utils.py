import mujoco
import numpy as np

from .exceptions import InvalidKeyframe
from .lie import SE3, SO3

_TYPE_TO_ENUM = {
    "body": mujoco.mjtObj.mjOBJ_BODY,
    "geom": mujoco.mjtObj.mjOBJ_GEOM,
    "site": mujoco.mjtObj.mjOBJ_SITE,
}

_TYPE_TO_XMAT_ATTRIBUTE = {
    "body": "xmat",
    "geom": "geom_xmat",
    "site": "site_xmat",
}

_TYPE_TO_POS_ATTRIBUTE = {
    "body": "xpos",
    "geom": "geom_xpos",
    "site": "site_xpos",
}


def dof_width(joint_type: int) -> int:
    """Get the dimensionality of the joint in qvel."""
    return {0: 6, 1: 3, 2: 1, 3: 1}[joint_type]


def qpos_width(joint_type: int) -> int:
    """Get the dimensionality of the joint in qpos."""
    return {0: 7, 1: 4, 2: 1, 3: 1}[joint_type]


def set_mocap_pose_from_site(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mocap_name: str,
    site_name: str,
):
    mocap_id = model.body(mocap_name).mocapid[0]
    site_id = data.site(site_name).id
    data.mocap_pos[mocap_id] = data.site_xpos[site_id]
    mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], data.site_xmat[site_id])


def set_mocap_pose_from_geom(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mocap_name: str,
    geom_name: str,
):
    mocap_id = model.body(mocap_name).mocapid[0]
    geom_id = data.geom(geom_name).id
    data.mocap_pos[mocap_id] = data.geom_xpos[geom_id]
    mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], data.geom_xmat[geom_id])


def set_mocap_pose_from_body(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mocap_name: str,
    body_name: str,
):
    mocap_id = model.body(mocap_name).mocapid[0]
    body_id = data.body(body_name).id
    data.mocap_pos[mocap_id] = data.xpos[body_id]
    mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], data.xmat[body_id])


def set_mocap_pose_from_obj(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mocap_name: str,
    obj_name: str,
    obj_type: str,
):
    mocap_id = model.body(mocap_name).mocapid[0]
    obj_id = mujoco.mj_name2id(model, _TYPE_TO_ENUM[obj_name], obj_name)
    data.mocap_pos[mocap_id] = getattr(data, _TYPE_TO_POS_ATTRIBUTE[obj_type])[obj_id]
    mujoco.mju_mat2Quat(
        data.mocap_quat[mocap_id],
        getattr(data, _TYPE_TO_XMAT_ATTRIBUTE[obj_type])[obj_id],
    )


def pose_from_mocap(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mocap_name: str,
) -> SE3:
    mocap_id = model.body(mocap_name).mocapid[0]
    return SE3.from_rotation_and_translation(
        rotation=SO3(data.mocap_quat[mocap_id]),
        translation=data.mocap_pos[mocap_id],
    )


def get_freejoint_dims(model: mujoco.MjModel) -> tuple[list[int], list[int]]:
    """Get all floating joint configuration and tangent indices."""
    q_ids = []
    v_ids = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            qadr = model.jnt_qposadr[j]
            vadr = model.jnt_dofadr[j]
            q_ids.extend(range(qadr, qadr + 7))
            v_ids.extend(range(vadr, vadr + 6))
    return q_ids, v_ids


def custom_configuration_vector(
    model: mujoco.MjModel,
    key_name: str | None = None,
    **kwargs,
) -> np.ndarray:
    """Generate a configuration vector where named joints have specific values.

    Args:
        model: An MjModel instance.
        key_name: Optional keyframe name to initialize the configuration vector from.
            Otherwise, the default pose qpos0 is used.
        kwargs: Custom values for joint coordinates.

    Returns:
        Configuration vector where named joints have the values specified in
            keyword arguments, and other joints have their neutral value or value
            defined in the keyframe if provided.
    """
    data = mujoco.MjData(model)
    if key_name is not None:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        if key_id == -1:
            raise InvalidKeyframe(key_name, model)
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    q = data.qpos.copy()
    for name, value in kwargs.items():
        jid = model.joint(name).id
        jnt_type = model.jnt_type[jid]
        jnt_dim = qpos_width(jnt_type)
        qid = model.jnt_dofadr[jid]
        value = np.atleast_1d(value)
        if value.shape != (jnt_dim,):
            raise ValueError(
                f"Joint {name} should have a qpos value of {jnt_dim,} but "
                f"got {value.shape}"
            )
        q[qid : qid + jnt_dim] = value
    return q
