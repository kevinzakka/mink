from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "apptronik_apollo" / "scene_table.xml"


def compute_look_at_rotation(
    source_pos: np.ndarray,
    target_pos: np.ndarray,
    world_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> mink.SE3:
    """Rotation whose +x axis points from `source_pos` toward `target_pos`, with +z aligned to `world_up`."""
    eps = 1e-6
    forward = target_pos - source_pos
    norm = np.linalg.norm(forward)
    if norm < eps:
        return mink.SE3.identity()
    forward = forward / norm

    up_ref = world_up
    if abs(np.dot(forward, up_ref)) > 1.0 - eps:
        up_ref = np.array([1.0, 0.0, 0.0])

    left = np.cross(up_ref, forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)

    rot = mink.SO3.from_matrix(np.column_stack((forward, left, up)))
    return mink.SE3.from_rotation(rot)


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)

    feet = ["left_foot", "right_foot"]
    hands = ["left_palm", "right_palm"]

    tasks = [
        pelvis_orientation_task := mink.FrameTask(
            frame_name="base_link",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        torso_orientation_task := mink.FrameTask(
            frame_name="torso_link",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model, cost=1e-1),
        com_task := mink.ComTask(cost=10.0),
    ]

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=100.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        feet_tasks.append(task)
    tasks.extend(feet_tasks)

    hand_tasks = []
    for hand in hands:
        task = mink.FrameTask(
            frame_name=hand,
            frame_type="site",
            position_cost=5.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        hand_tasks.append(task)
    tasks.extend(hand_tasks)

    head_task = mink.FrameTask(
        frame_name="head",
        frame_type="site",
        position_cost=0.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    tasks.append(head_task)

    # Enable collision avoidance between the following geoms.
    # left hand - table, right hand - table
    # left hand - left thigh, right hand - right thigh
    collision_pairs = [
        (["collision_r_hand_plate", "collision_l_hand_plate"], ["table"]),
        (["collision_r_hand_plate"], ["collision_capsule_body_r_thigh"]),
        (["collision_l_hand_plate"], ["collision_capsule_body_l_thigh"]),
    ]
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,  # type: ignore
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.1,
    )
    limits = [
        mink.ConfigurationLimit(model),
        collision_avoidance_limit,
    ]

    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
    hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in hands]
    head_mid = model.body("head_target").mocapid[0]
    model = configuration.model
    data = configuration.data
    solver = "daqp"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("teleop")
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)
        torso_orientation_task.set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective sites.
        for hand, foot in zip(hands, feet):
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
        data.mocap_pos[com_mid] = data.subtree_com[1]  # Hardcoded index 1.

        data.mocap_pos[head_mid] = data.mocap_pos[com_mid] + np.array([1.0, 0.0, 0.5])

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update task targets.
            com_task.set_target(data.mocap_pos[com_mid])
            for i, (hand_task, foot_task) in enumerate(zip(hand_tasks, feet_tasks)):
                foot_task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))
                hand_task.set_target(mink.SE3.from_mocap_id(data, hands_mid[i]))

            head_target = compute_look_at_rotation(
                source_pos=data.site_xpos[data.site("head").id],
                target_pos=data.mocap_pos[head_mid],
            )
            head_task.set_target(head_target)

            # Compute velocity and integrate into the next configuration.
            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, damping=1e-1, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Note the below are optional: they are used to visualize the output of the
            # fromto sensor which is used by the collision avoidance constraint.
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
