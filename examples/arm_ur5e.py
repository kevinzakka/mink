import mujoco
import mujoco.viewer
import numpy as np
import mink
from pathlib import Path
import time
import qpsolvers

_HERE = Path(__file__).resolve().parent
_XML_PATH = _HERE / "universal_robots_ur5e" / "scene.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(_XML_PATH.as_posix())
    data = mujoco.MjData(model)

    dt = 0.002
    model.opt.timestep = dt

    model.body_gravcomp[:] = 1.0

    joints = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]

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
        limit_gain=0.5,
    )

    velocity_limit = mink.VelocityLimit.initialize(
        model=model,
        joint2limit={
            "shoulder_pan": np.pi,
            "shoulder_lift": np.pi,
            "elbow": np.pi,
            "wrist_1": np.pi,
            "wrist_2": np.pi,
            "wrist_3": np.pi,
        },
    )

    limits = [
        configuration_limit,
        velocity_limit,
    ]

    keyframe_name = "home"
    configuration = mink.Configuration.initialize_from_keyframe(
        model=model, data=data, keyframe_name=keyframe_name
    )

    for task in tasks:
        task.set_target_from_configuration(configuration)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "quadprog" in qpsolvers.available_solvers:
        solver = "quadprog"
    print(f"Using {solver} solver")

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, model.key(keyframe_name).id)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()

            new_end_effector_target = mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(wxyz=data.mocap_quat[0]),
                translation=data.mocap_pos[0],
            )
            end_effector_task.set_target(new_end_effector_target)

            dq = mink.solve_ik(
                configuration=configuration,
                tasks=tasks,
                limits=limits,
                dt=dt,
                solver=solver,
            )

            q = configuration.integrate(dq, dt)
            np.clip(q, *model.jnt_range.T, out=data.ctrl)
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
