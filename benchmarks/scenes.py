"""Canonical IK benchmark scenarios shared by every benchmark script.

A Scene bundles two callables. setup() builds the mink objects (configuration,
tasks, limits) and returns a state dict; the state must contain the keys
configuration, tasks, limits and damping, and may stash extra bookkeeping (mocap
ids, initial poses) alongside them. update(state, t) moves the task targets for
simulation time t in seconds but does not solve, so the same scene can be driven
by an end-to-end timer, a per-component profiler, or anything else, all measuring
the same problem.

Adding a scenario (e.g. a recorded marionette trajectory) is a new setup/update
pair plus one entry in SCENES.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

import mink
from mink.lie import SE3

_HERE = Path(__file__).parent
_EXAMPLES = _HERE.parent / "examples"

#: Default IK integration timestep [s] (500 Hz).
DT = 1.0 / 500.0

State = dict
SetupFn = Callable[[], State]
UpdateFn = Callable[[State, float], None]


@dataclass(frozen=True)
class Scene:
    """A named IK benchmark scenario."""

    key: str
    label: str
    nv: int  # Tangent-space dimension, for display only.
    setup: SetupFn
    update: UpdateFn


# ---------------------------------------------------------------------------
# UR5e: single 6-DoF arm, one frame task + configuration and velocity limits.
# ---------------------------------------------------------------------------

_UR5E_XML = _EXAMPLES / "universal_robots_ur5e" / "scene.xml"


def setup_ur5e() -> State:
    model = mujoco.MjModel.from_xml_path(_UR5E_XML.as_posix())
    configuration = mink.Configuration(model)
    model = configuration.model
    data = configuration.data

    ee_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1e-6,
    )
    tasks = [ee_task]
    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(
            model,
            {
                name: np.pi
                for name in [
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow",
                    "wrist_1",
                    "wrist_2",
                    "wrist_3",
                ]
            },
        ),
    ]

    configuration.update_from_keyframe("home")
    mid = model.body("target").mocapid[0]
    mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
    ee_task.set_target(mink.SE3.from_mocap_id(data, mid))

    return dict(
        configuration=configuration,
        data=data,
        tasks=tasks,
        limits=limits,
        damping=1e-6,
        ee_task=ee_task,
        mid=mid,
        init_pos=data.mocap_pos[mid].copy(),
    )


def update_ur5e(s: State, t: float) -> None:
    data = s["data"]
    angle = 2.0 * np.pi * 0.5 * t
    pos = s["init_pos"].copy()
    pos[0] += 0.08 * np.cos(angle)
    pos[2] += 0.08 * np.sin(angle)
    data.mocap_pos[s["mid"]] = pos
    s["ee_task"].set_target(mink.SE3.from_mocap_id(data, s["mid"]))


# ---------------------------------------------------------------------------
# G1 humanoid: 49-DoF floating base, 8 tasks + light collision avoidance.
# ---------------------------------------------------------------------------

_G1_XML = _EXAMPLES / "unitree_g1" / "scene.xml"


def setup_humanoid() -> State:
    model = mujoco.MjModel.from_xml_path(_G1_XML.as_posix())
    configuration = mink.Configuration(model)
    model = configuration.model
    data = configuration.data

    feet = ["right_foot", "left_foot"]
    pelvis_task = mink.FrameTask(
        "pelvis", "body", position_cost=0.0, orientation_cost=10.0, lm_damping=1e-3
    )
    torso_task = mink.FrameTask(
        "torso_link", "body", position_cost=0.0, orientation_cost=5.0, lm_damping=1e-3
    )
    posture_task = mink.PostureTask(model, cost=1e-1)
    com_task = mink.ComTask(cost=50.0)
    feet_tasks = [
        mink.FrameTask(
            f, "site", position_cost=100.0, orientation_cost=10.0, lm_damping=0.0
        )
        for f in feet
    ]
    r_hand_task = mink.FrameTask(
        "right_palm",
        "site",
        position_cost=5.0,
        orientation_cost=0.5,
        gain=0.5,
        lm_damping=1e-3,
    )
    relative_task = mink.RelativeFrameTask(
        "left_palm",
        "site",
        root_name="torso_link",
        root_type="body",
        position_cost=5.0,
        orientation_cost=0.5,
        gain=0.5,
        lm_damping=1e-3,
    )
    tasks = [
        pelvis_task,
        torso_task,
        posture_task,
        com_task,
        *feet_tasks,
        r_hand_task,
        relative_task,
    ]

    collision_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=[
            (["left_hand_collision"], ["left_thigh_collision"]),
            (["right_hand_collision"], ["right_thigh_collision"]),
        ],
        minimum_distance_from_collisions=0.005,
        collision_detection_distance=0.15,
    )
    limits = [mink.ConfigurationLimit(model), collision_limit]

    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{f}_target").mocapid[0] for f in feet]
    r_hand_mid = model.body("right_palm_target").mocapid[0]

    configuration.update_from_keyframe("stand")
    posture_task.set_target_from_configuration(configuration)
    pelvis_task.set_target_from_configuration(configuration)
    torso_task.set_target_from_configuration(configuration)
    for f in feet:
        mink.move_mocap_to_frame(model, data, f"{f}_target", f, "site")
    mink.move_mocap_to_frame(model, data, "right_palm_target", "right_palm", "site")
    data.mocap_pos[com_mid] = data.subtree_com[1].copy()
    relative_task.set_target_from_configuration(configuration)

    return dict(
        configuration=configuration,
        data=data,
        tasks=tasks,
        limits=limits,
        damping=1e-2,
        com_task=com_task,
        feet_tasks=feet_tasks,
        r_hand_task=r_hand_task,
        relative_task=relative_task,
        com_mid=com_mid,
        feet_mid=feet_mid,
        r_hand_mid=r_hand_mid,
        init_com=data.subtree_com[1].copy(),
        init_r_hand_pos=data.mocap_pos[r_hand_mid].copy(),
        init_relative_target=relative_task.transform_target_to_root,
    )


def update_humanoid(s: State, t: float) -> None:
    data = s["data"]

    squat = 0.5 * (1.0 - np.cos(2.0 * np.pi * 0.5 * t))
    com_target = s["init_com"].copy()
    com_target[2] -= 0.12 * squat
    data.mocap_pos[s["com_mid"]] = com_target

    angle = 2.0 * np.pi * 0.8 * t
    r_pos = s["init_r_hand_pos"].copy()
    r_pos[0] += 0.04 * np.cos(angle)
    r_pos[2] += 0.04 * np.sin(angle)
    data.mocap_pos[s["r_hand_mid"]] = r_pos

    wave_t = 2.0 * np.pi * 0.8 * t
    target_in_root = s["init_relative_target"] @ SE3.from_translation(
        np.array([0.0, 0.05 + 0.04 * np.sin(wave_t), 0.04 * np.cos(wave_t)])
    )
    s["relative_task"].set_target(target_in_root)
    s["com_task"].set_target(data.mocap_pos[s["com_mid"]])
    for i, ft in enumerate(s["feet_tasks"]):
        ft.set_target(mink.SE3.from_mocap_id(data, s["feet_mid"][i]))
    s["r_hand_task"].set_target(mink.SE3.from_mocap_id(data, s["r_hand_mid"]))


# ---------------------------------------------------------------------------
# ALOHA: dual-arm, two frame tasks + dense collision avoidance (1104 pairs).
# ---------------------------------------------------------------------------

_ALOHA_XML = _EXAMPLES / "aloha" / "scene.xml"
_ALOHA_JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]


def setup_aloha() -> State:
    model = mujoco.MjModel.from_xml_path(_ALOHA_XML.as_posix())
    data = mujoco.MjData(model)
    configuration = mink.Configuration(model)

    velocity_limits = {
        f"{p}/{n}": np.pi for p in ["left", "right"] for n in _ALOHA_JOINT_NAMES
    }
    l_ee_task = mink.FrameTask(
        "left/gripper", "site", position_cost=5.0, orientation_cost=1.0, lm_damping=1e-3
    )
    r_ee_task = mink.FrameTask(
        "right/gripper",
        "site",
        position_cost=5.0,
        orientation_cost=1.0,
        lm_damping=1e-3,
    )
    tasks = [l_ee_task, r_ee_task]

    l_wrist = mink.get_subtree_geom_ids(model, model.body("left/wrist_link").id)
    r_wrist = mink.get_subtree_geom_ids(model, model.body("right/wrist_link").id)
    l_geoms = mink.get_subtree_geom_ids(model, model.body("left/upper_arm_link").id)
    r_geoms = mink.get_subtree_geom_ids(model, model.body("right/upper_arm_link").id)
    frame_geoms = mink.get_body_geom_ids(model, model.body("metal_frame").id)
    collision_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=[(l_wrist, r_wrist), (l_geoms + r_geoms, frame_geoms + ["table"])],
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.1,
    )
    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, velocity_limits),
        collision_limit,
    ]

    l_mid = model.body("left/target").mocapid[0]
    r_mid = model.body("right/target").mocapid[0]
    mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
    configuration.update(data.qpos)
    mujoco.mj_forward(model, data)
    mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
    mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")

    return dict(
        configuration=configuration,
        data=data,
        tasks=tasks,
        limits=limits,
        damping=1e-4,
        l_ee_task=l_ee_task,
        r_ee_task=r_ee_task,
        l_mid=l_mid,
        r_mid=r_mid,
        init_l_pos=data.mocap_pos[l_mid].copy(),
        init_r_pos=data.mocap_pos[r_mid].copy(),
        init_l_quat=data.mocap_quat[l_mid].copy(),
        init_r_quat=data.mocap_quat[r_mid].copy(),
    )


def update_aloha(s: State, t: float) -> None:
    data = s["data"]
    phase = 2.0 * np.pi * 0.4 * t
    inward = 0.5 * (1.0 - np.cos(phase))

    l_pos = s["init_l_pos"].copy()
    l_pos[0] += 0.03 + (0.16 - 0.03) * inward
    l_pos[1] += -0.04 * (1.0 - inward)
    l_pos[2] += -0.24 * inward

    r_pos = s["init_r_pos"].copy()
    r_pos[0] -= 0.03 + (0.16 - 0.03) * inward
    r_pos[1] += -0.04 * (1.0 - inward)
    r_pos[2] += -0.24 * inward

    data.mocap_pos[s["l_mid"]] = l_pos
    data.mocap_pos[s["r_mid"]] = r_pos
    data.mocap_quat[s["l_mid"]] = s["init_l_quat"]
    data.mocap_quat[s["r_mid"]] = s["init_r_quat"]
    s["l_ee_task"].set_target(mink.SE3.from_mocap_id(data, s["l_mid"]))
    s["r_ee_task"].set_target(mink.SE3.from_mocap_id(data, s["r_mid"]))


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------

SCENES: dict[str, Scene] = {
    s.key: s
    for s in [
        Scene("ur5e", "UR5e arm (1 frame task + limits)", 6, setup_ur5e, update_ur5e),
        Scene(
            "humanoid",
            "G1 humanoid (8 tasks + collision)",
            49,
            setup_humanoid,
            update_humanoid,
        ),
        Scene(
            "aloha",
            "ALOHA dual-arm (1104 collision pairs)",
            12,
            setup_aloha,
            update_aloha,
        ),
    ]
}
