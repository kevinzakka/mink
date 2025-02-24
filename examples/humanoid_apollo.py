from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import numpy as np

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "apptronik_apollo" / "scene.xml"


def compute_look_at_rotation(
    head_pos: np.ndarray, target_pos: np.ndarray, world_up=np.array([0.0, 0.0, 1.0])
):
    # Compute look direction.
    look_direction = target_pos - head_pos
    x_axis = look_direction / np.linalg.norm(look_direction)

    # Compute the intermediate y-axis using the world up vector.
    y_axis = np.cross(world_up, x_axis)
    norm_y = np.linalg.norm(y_axis)
    if norm_y < 1e-6:
        # The look_direction is nearly parallel to world_up; choose an arbitrary vector.
        y_axis = np.cross(x_axis, np.array([1.0, 0.0, 0.0]))
        norm_y = np.linalg.norm(y_axis)
        if norm_y < 1e-6:
            y_axis = np.cross(x_axis, np.array([0.0, 1.0, 0.0]))
            norm_y = np.linalg.norm(y_axis)
    y_axis /= norm_y

    z_axis = np.cross(x_axis, y_axis)

    rot = mink.SO3.from_matrix(np.column_stack((x_axis, y_axis, z_axis)))
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
            orientation_cost=10.0,
        ),
        torso_orientation_task := mink.FrameTask(
            frame_name="torso_link",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=10.0,
        ),
        posture_task := mink.PostureTask(model, cost=1.0),
        com_task := mink.ComTask(cost=200.0),
    ]

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        )
        feet_tasks.append(task)
    tasks.extend(feet_tasks)

    hand_tasks = []
    for hand in hands:
        task = mink.FrameTask(
            frame_name=hand,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        hand_tasks.append(task)
    tasks.extend(hand_tasks)

    head_task = mink.FrameTask(
        frame_name="head",
        frame_type="site",
        position_cost=0.0,
        orientation_cost=10.0,
        lm_damping=1.0,
    )
    tasks.append(head_task)

    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
    hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in hands]
    head_mid = model.body("head_target").mocapid[0]
    model = configuration.model
    data = configuration.data
    solver = "quadprog"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("stand")
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)
        torso_orientation_task.set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective sites.
        for hand, foot in zip(hands, feet):
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
        data.mocap_pos[com_mid] = data.subtree_com[1]

        data.mocap_pos[head_mid] = data.mocap_pos[com_mid] + np.array([1.0, 0.0, 0.5])

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update task targets.
            com_task.set_target(data.mocap_pos[com_mid])
            for i, (hand_task, foot_task) in enumerate(zip(hand_tasks, feet_tasks)):
                foot_task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))
                hand_task.set_target(mink.SE3.from_mocap_id(data, hands_mid[i]))

            head_target = compute_look_at_rotation(
                head_pos=data.site_xpos[data.site("head").id],
                target_pos=data.mocap_pos[head_mid],
            )
            head_task.set_target(head_target)

            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-1)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
