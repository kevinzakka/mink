import mujoco
import mujoco.viewer
import mink
from mink import lie
import time
from pathlib import Path

_HERE = Path(__file__).parent
_XML = _HERE / "unitree_go1" / "scene.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    dt = 0.002
    model.opt.timestep = dt

    keyframe_name = "home"
    configuration = mink.Configuration.initialize_from_keyframe(
        model=model, data=data, keyframe_name=keyframe_name
    )

    joints = [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]

    feet = ["FL", "FR", "RR", "RL"]

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

    base_task = mink.FrameTask.initialize(
        frame_name="trunk",
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    posture_task = mink.PostureTask.initialize(cost=1e-5)

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask.initialize(
            frame_name=foot,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1e-5,
        )
        feet_tasks.append(task)

    tasks = [base_task, posture_task, *feet_tasks]

    for task in tasks:
        task.set_target_from_configuration(configuration)

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        posture_task.set_target(data.qpos.copy())

        for foot in feet:
            data.mocap_pos[model.body(f"{foot}_target").mocapid[0]] = data.site(
                foot
            ).xpos.copy()
            mujoco.mju_mat2Quat(
                data.mocap_quat[model.body(f"{foot}_target").mocapid[0]],
                data.site_xmat[data.site(foot).id].copy(),
            )

        data.mocap_pos[0] = data.xpos[model.body("trunk").id].copy()
        mujoco.mju_mat2Quat(
            data.mocap_quat[0], data.xmat[model.body("trunk").id].copy()
        )

        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        while viewer.is_running():
            step_start = time.time()

            base_target = lie.SE3.from_rotation_and_translation(
                rotation=lie.SO3(data.mocap_quat[0]),
                translation=data.mocap_pos[0],
            )
            base_task.set_target(base_target)

            for i, foot in enumerate(feet):
                foot_target = lie.SE3.from_rotation_and_translation(
                    rotation=lie.SO3(
                        data.mocap_quat[model.body(f"{foot}_target").mocapid[0]]
                    ),
                    translation=data.mocap_pos[model.body(f"{foot}_target").mocapid[0]],
                )
                feet_tasks[i].set_target(foot_target)

            velocity = mink.solve_ik(
                configuration=configuration,
                tasks=tasks,
                limits=[],
                dt=dt,
            )
            print(velocity)
            data.ctrl = configuration.integrate(velocity, dt)[7:]
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
