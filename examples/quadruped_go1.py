import mujoco
import mujoco.viewer
import mink
from pathlib import Path
import numpy as np
from loop_rate_limiters import RateLimiter
from mink.utils import set_mocap_pose_from_site, set_mocap_pose_from_body

_HERE = Path(__file__).parent
_XML = _HERE / "unitree_go1" / "scene.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    feet = ["FL", "FR", "RR", "RL"]

    limits = [
        mink.ConfigurationLimit(model=model),
    ]

    #
    # Tasks.
    #

    base_task = mink.FrameTask(
        frame_name="trunk",
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    posture_task = mink.PostureTask(cost=1e-5)

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
        )
        feet_tasks.append(task)

    tasks = [base_task, posture_task, *feet_tasks]

    configuration = mink.Configuration(model)
    for task in tasks:
        task.set_target_from_configuration(configuration)

    model = configuration.model
    data = configuration.data
    velocity = None

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Initialize to the home keyframe.
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update()
        posture_task.set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective targetees.
        for foot in feet:
            set_mocap_pose_from_site(model, data, f"{foot}_target", foot)
        set_mocap_pose_from_body(model, data, "trunk_target", "trunk")

        # Initialize the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        rate = RateLimiter(frequency=500.0)
        dt = rate.period
        sim_steps_per_control_steps = int(np.ceil(dt / model.opt.timestep))
        while viewer.is_running():
            # Update task targets.
            base_task.set_target_from_mocap(data, 0)
            for i, task in enumerate(feet_tasks):
                mocap_id = model.body(f"{feet[i]}_target").mocapid[0]
                task.set_target_from_mocap(data, mocap_id)

            # Compute velocity, integrate and set control signal.
            velocity = mink.solve_ik(configuration, tasks, limits, dt, 1e-5, velocity)
            data.ctrl = configuration.integrate(velocity, dt)[7:]
            mujoco.mj_step(model, data, sim_steps_per_control_steps)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
