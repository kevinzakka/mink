import mujoco
import mujoco.viewer
import mink
from mink import lie
import time
from pathlib import Path

_HERE = Path(__file__).parent
_XML = _HERE / "universal_robots_ur5e" / "scene.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    dt = 0.002
    model.opt.timestep = dt

    joints = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]

    keyframe_name = "home"
    configuration = mink.Configuration.initialize_from_keyframe(
        model=model, data=data, keyframe_name=keyframe_name
    )

    #
    # Tasks.
    #

    end_effector_task = mink.FrameTask.initialize(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    tasks = [
        end_effector_task,
    ]

    #
    # Limits.
    #

    configuration_limit = mink.ConfigurationLimit.initialize(
        model=model,
        joints=joints,
        limit_gain=0.95,
    )

    limits = [
        configuration_limit,
    ]

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        data.mocap_pos[0] = data.site("attachment_site").xpos.copy()
        mujoco.mju_mat2Quat(
            data.mocap_quat[0], data.site("attachment_site").xmat.copy()
        )

        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()

            end_effector_target = lie.SE3.from_rotation_and_translation(
                rotation=lie.SO3(data.mocap_quat[0]),
                translation=data.mocap_pos[0],
            )
            end_effector_task.set_target(end_effector_target)

            velocity = mink.solve_ik(
                configuration=configuration,
                tasks=tasks,
                limits=limits,
                dt=dt,
            )

            data.ctrl = configuration.integrate(velocity, dt)
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
