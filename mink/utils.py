import mujoco
import numpy as np

from .lie import SE3, SO3


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
    **kwargs,
) -> np.ndarray:
    raise NotImplementedError


def dof_width(joint_type: int) -> int:
    return {0: 6, 1: 3, 2: 1, 3: 1}[joint_type]


def qpos_width(joint_type: int) -> int:
    return {0: 7, 1: 4, 2: 1, 3: 1}[joint_type]
