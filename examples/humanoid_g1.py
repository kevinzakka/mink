import mujoco
import mujoco.viewer
import mink
from mink import lie
import time
import numpy as np
from pathlib import Path

_HERE = Path(__file__).parent
_XML = _HERE / "unitree_g1" / "scene.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    dt = 0.002
    model.opt.timestep = dt

    keyframe_name = "stand"
    configuration = mink.Configuration.initialize_from_keyframe(
        model=model, data=data, keyframe_name=keyframe_name
    )

    joints = [model.joint(i).name for i in range(1, model.njnt)]
    feet = ["right_foot", "left_foot"]
    hands = ["right_palm", "left_palm"]

    #
    # Limits.
    #

    configuration_limit = mink.ConfigurationLimit.initialize(
        model=model,
        joints=joints,
        limit_gain=0.5,
    )

    limits = [
        configuration_limit,
    ]

    #
    # Tasks.
    #

    pelvis_orientation_task = mink.FrameTask.initialize(
        frame_name="pelvis",
        frame_type="body",
        position_cost=0.0,
        orientation_cost=10.0,
    )
    com_task = mink.ComTask.initialize(cost=200.0)

    posture_task = mink.PostureTask.initialize(cost=1e-1)

    tasks = [pelvis_orientation_task, posture_task, com_task]

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask.initialize(
            frame_name=foot,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=10.0,
        )
        feet_tasks.append(task)
    tasks.extend(feet_tasks)

    hand_tasks = []
    for hand in hands:
        task = mink.FrameTask.initialize(
            frame_name=hand,
            frame_type="site",
            position_cost=4.0,
            orientation_cost=0.0,
        )
        hand_tasks.append(task)
    tasks.extend(hand_tasks)

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)

        for task in tasks:
            task.set_target_from_configuration(configuration)
        posture_task.set_target(data.qpos)
        for hand in hands:
            data.mocap_pos[model.body(f"{hand}_target").mocapid[0]] = data.site(hand).xpos.copy()
            mujoco.mju_mat2Quat(data.mocap_quat[model.body(f"{hand}_target").mocapid[0]], data.site_xmat[data.site(hand).id].copy())
        for foot in feet:
            data.mocap_pos[model.body(f"{foot}_target").mocapid[0]] = data.site(foot).xpos.copy()
            mujoco.mju_mat2Quat(data.mocap_quat[model.body(f"{foot}_target").mocapid[0]], data.site_xmat[data.site(foot).id].copy())

        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        data.mocap_pos[0] = data.subtree_com[1].copy()

        while viewer.is_running():
            step_start = time.time()

            for i, hand in enumerate(hands):
                hand_target = lie.SE3.from_rotation_and_translation(
                    rotation=lie.SO3(
                        data.mocap_quat[model.body(f"{hand}_target").mocapid[0]]
                    ),
                    translation=data.mocap_pos[model.body(f"{hand}_target").mocapid[0]],
                )
                hand_tasks[i].set_target(hand_target)
            for i, foot in enumerate(feet):
                foot_target = lie.SE3.from_rotation_and_translation(
                    rotation=lie.SO3(
                        data.mocap_quat[model.body(f"{foot}_target").mocapid[0]]
                    ),
                    translation=data.mocap_pos[model.body(f"{foot}_target").mocapid[0]],
                )
                feet_tasks[i].set_target(foot_target)

            com_task.set_target(data.mocap_pos[0])

            velocity = mink.solve_ik(
                configuration=configuration,
                tasks=tasks,
                limits=[],
                dt=dt,
            )
            configuration.integrate_in_place(velocity, dt)
            mujoco.mj_forward(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
