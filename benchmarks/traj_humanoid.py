"""Humanoid G1 trajectory: hands trace patterns while squatting.

Feet pinned, COM drives squat, right hand traces world-space circle,
left hand traces torso-relative circle via RelativeFrameTask,
collision avoidance between hands and hips.
"""

import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink
from mink.lie import SE3

_HERE = Path(__file__).parent
_XML = _HERE.parent / "examples" / "unitree_g1" / "scene.xml"
sys.path.insert(0, str(_HERE.parent))
from examples._model_assets import load_example_spec  # noqa: E402

if __name__ == "__main__":
    model = load_example_spec("unitree_g1", _XML).compile()

    configuration = mink.Configuration(model)
    feet = ["right_foot", "left_foot"]

    # === Task stack (priority descending by cost) ===

    tasks = [
        # Pelvis orientation: keep upright. No position control (floating base).
        pelvis_orientation_task := mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=10.0,
            lm_damping=1e-3,
        ),
        # Torso orientation: less critical than pelvis.
        torso_orientation_task := mink.FrameTask(
            frame_name="torso_link",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=5.0,
            lm_damping=1e-3,
        ),
        # Posture: null-space regularization only. Low cost so it never
        # interferes with higher-priority tasks.
        posture_task := mink.PostureTask(model, cost=1e-1),
        # COM: balance-critical, always feasible since we control the target.
        com_task := mink.ComTask(cost=50.0),
    ]

    # Feet: support base, highest priority. Zero damping — targets are the
    # initial positions which are always feasible.
    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=100.0,
            orientation_cost=10.0,
            lm_damping=0.0,
        )
        feet_tasks.append(task)
    tasks.extend(feet_tasks)

    # Right hand: world-space circle via FrameTask.
    r_hand_task = mink.FrameTask(
        frame_name="right_palm",
        frame_type="site",
        position_cost=5.0,
        orientation_cost=0.5,
        gain=0.5,
        lm_damping=1e-3,
    )
    tasks.append(r_hand_task)

    # Left hand: torso-relative circle via RelativeFrameTask.
    relative_task = mink.RelativeFrameTask(
        frame_name="left_palm",
        frame_type="site",
        root_name="torso_link",
        root_type="body",
        position_cost=5.0,
        orientation_cost=0.5,
        gain=0.5,
        lm_damping=1e-3,
    )
    tasks.append(relative_task)

    # === Collision avoidance ===

    collision_pairs = [
        (["left_hand_collision"], ["left_thigh_collision"]),
        (["right_hand_collision"], ["right_thigh_collision"]),
    ]
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,  # type: ignore
        minimum_distance_from_collisions=0.005,
        collision_detection_distance=0.15,
    )
    limits = [mink.ConfigurationLimit(model), collision_avoidance_limit]

    # === Mocap IDs ===

    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
    r_hand_mid = model.body("right_palm_target").mocapid[0]
    l_hand_mid = model.body("left_palm_target").mocapid[0]

    model = configuration.model
    data = configuration.data
    solver = "daqp"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        configuration.update_from_keyframe("stand")
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)
        torso_orientation_task.set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective sites.
        for foot in feet:
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
        mink.move_mocap_to_frame(model, data, "right_palm_target", "right_palm", "site")
        mink.move_mocap_to_frame(model, data, "left_palm_target", "left_palm", "site")
        data.mocap_pos[com_mid] = data.subtree_com[1].copy()

        relative_task.set_target_from_configuration(configuration)
        assert relative_task.transform_target_to_root is not None
        init_relative_target = relative_task.transform_target_to_root

        # Record initial positions.
        init_com = data.subtree_com[1].copy()
        init_r_hand_pos = data.mocap_pos[r_hand_mid].copy()

        # === Trajectory parameters ===
        squat_depth = 0.12
        squat_freq = 0.5  # one squat every 2s
        circle_radius = 0.04
        circle_freq = 0.8
        wave_radius = 0.04
        wave_freq = 0.8
        wave_y_offset = 0.05

        rate = RateLimiter(frequency=500.0, warn=False)
        t = 0.0
        while viewer.is_running():
            # --- Squat: lower and raise COM sinusoidally ---
            squat_phase = 0.5 * (1.0 - np.cos(2.0 * np.pi * squat_freq * t))
            com_target = init_com.copy()
            com_target[2] -= squat_depth * squat_phase
            data.mocap_pos[com_mid] = com_target

            # --- Right hand: world-space circle in the sagittal plane ---
            angle = 2.0 * np.pi * circle_freq * t
            r_pos = init_r_hand_pos.copy()
            r_pos[0] += circle_radius * np.cos(angle)
            r_pos[2] += circle_radius * np.sin(angle)
            data.mocap_pos[r_hand_mid] = r_pos

            # --- Left hand: circle in frontal plane, torso-relative ---
            wave_t = 2.0 * np.pi * wave_freq * t
            dy = wave_y_offset + wave_radius * np.sin(wave_t)
            dz = wave_radius * np.cos(wave_t)
            target_in_root = init_relative_target @ SE3.from_translation(
                np.array([0.0, dy, dz])
            )
            relative_task.set_target(target_in_root)

            # Visualize the relative task target in world space.
            torso_in_world = configuration.get_transform_frame_to_world(
                "torso_link", "body"
            )
            target_in_world = torso_in_world @ target_in_root
            data.mocap_pos[l_hand_mid] = target_in_world.translation()
            data.mocap_quat[l_hand_mid] = target_in_world.rotation().wxyz

            # --- Update task targets ---
            com_task.set_target(data.mocap_pos[com_mid])
            for i, foot_task in enumerate(feet_tasks):
                foot_task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))
            r_hand_task.set_target(mink.SE3.from_mocap_id(data, r_hand_mid))

            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, damping=1e-2, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            viewer.sync()
            rate.sleep()
            t += rate.dt
