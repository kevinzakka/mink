"""ALOHA dual-arm trajectory: arms sweep toward each other and dip near table.

Both grippers trace mirrored elliptical paths that overlap at the center,
forcing collision avoidance between wrists. The paths dip near the table
surface, activating arm-table and arm-frame collision avoidance.
"""

import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE.parent / "examples" / "aloha" / "scene.xml"
sys.path.insert(0, str(_HERE.parent))
from examples._model_assets import load_example_spec  # noqa: E402

_JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]
_VELOCITY_LIMITS = {k: np.pi for k in _JOINT_NAMES}


if __name__ == "__main__":
    model = load_example_spec("aloha", _XML).compile()
    data = mujoco.MjData(model)

    joint_names: list[str] = []
    velocity_limits: dict[str, float] = {}
    for prefix in ["left", "right"]:
        for n in _JOINT_NAMES:
            name = f"{prefix}/{n}"
            joint_names.append(name)
            velocity_limits[name] = _VELOCITY_LIMITS[n]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    configuration = mink.Configuration(model)

    tasks = [
        l_ee_task := mink.FrameTask(
            frame_name="left/gripper",
            frame_type="site",
            position_cost=5.0,
            orientation_cost=1.0,
            gain=1.0,
            lm_damping=1e-3,
        ),
        r_ee_task := mink.FrameTask(
            frame_name="right/gripper",
            frame_type="site",
            position_cost=5.0,
            orientation_cost=1.0,
            gain=1.0,
            lm_damping=1e-3,
        ),
    ]

    # Collision avoidance: wrist-wrist + arms-table/frame.
    l_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("left/wrist_link").id)
    r_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("right/wrist_link").id)
    l_geoms = mink.get_subtree_geom_ids(model, model.body("left/upper_arm_link").id)
    r_geoms = mink.get_subtree_geom_ids(model, model.body("right/upper_arm_link").id)
    frame_geoms = mink.get_body_geom_ids(model, model.body("metal_frame").id)
    collision_pairs = [
        (l_wrist_geoms, r_wrist_geoms),
        (l_geoms + r_geoms, frame_geoms + ["table"]),
    ]
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.1,
    )

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, velocity_limits),
        collision_avoidance_limit,
    ]

    l_mid = model.body("left/target").mocapid[0]
    r_mid = model.body("right/target").mocapid[0]
    solver = "daqp"

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
        mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")

        init_l_pos = data.mocap_pos[l_mid].copy()
        init_r_pos = data.mocap_pos[r_mid].copy()
        init_l_quat = data.mocap_quat[l_mid].copy()
        init_r_quat = data.mocap_quat[r_mid].copy()

        # Trajectory: mirrored ellipses that cross at center and dip near table.
        # x-axis: inward/outward, z-axis: up/down.
        freq = 0.4  # Hz
        x_max = 0.16  # max inward sweep (arms try to cross → collision avoidance)
        x_min = 0.03  # min inward sweep (small pullback, never fully returns)
        z_dip = 0.24  # max downward dip from start (table collision avoidance)
        y_fwd = 0.04  # forward push when arms are wide

        rate = RateLimiter(frequency=500.0, warn=False)
        t = 0.0
        while viewer.is_running():
            phase = 2.0 * np.pi * freq * t

            # Smooth oscillation that starts at 0: (1 - cos(t)) / 2 → [0, 1]
            # x: oscillates between x_min and x_max inward.
            # z: dips down when arms are closest (phase offset by π/2).
            inward = 0.5 * (1.0 - np.cos(phase))  # 0 → 1 → 0
            x_sweep = x_min + (x_max - x_min) * inward
            z_drop = -z_dip * inward
            y_push = -y_fwd * (1.0 - inward)  # forward when arms are wide

            l_pos = init_l_pos.copy()
            l_pos[0] += x_sweep
            l_pos[1] += y_push
            l_pos[2] += z_drop

            r_pos = init_r_pos.copy()
            r_pos[0] -= x_sweep
            r_pos[1] += y_push
            r_pos[2] += z_drop

            data.mocap_pos[l_mid] = l_pos
            data.mocap_pos[r_mid] = r_pos
            data.mocap_quat[l_mid] = init_l_quat
            data.mocap_quat[r_mid] = init_r_quat

            l_ee_task.set_target(mink.SE3.from_mocap_id(data, l_mid))
            r_ee_task.set_target(mink.SE3.from_mocap_id(data, r_mid))

            vel = mink.solve_ik(
                configuration,
                tasks,
                rate.dt,
                solver,
                damping=1e-4,
                limits=limits,
            )
            configuration.integrate_inplace(vel, rate.dt)

            data.ctrl[actuator_ids] = configuration.q[dof_ids]
            mujoco.mj_step(model, data)

            viewer.sync()
            rate.sleep()
            t += rate.dt
