import mujoco
import mujoco.viewer
import numpy as np
import mink
from pathlib import Path
from loop_rate_limiters import RateLimiter

_HERE = Path(__file__).parent
_XML = _HERE / "universal_robots_ur5e" / "scene.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
    ]

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(np.deg2rad(180) * np.ones_like(configuration.q)),
    ]

    model = configuration.model
    data = configuration.data
    velocity = None

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Initialize to the home keyframe.
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update()

        # Initialize the mocap target at the end-effector site.
        data.mocap_pos[0] = data.site("attachment_site").xpos
        mujoco.mju_mat2Quat(data.mocap_quat[0], data.site("attachment_site").xmat)

        # Initialize the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        rate = RateLimiter(frequency=60.0)
        dt = rate.period
        sim_steps_per_control_steps = int(np.ceil(dt / model.opt.timestep))
        while viewer.is_running():
            # Update task target.
            end_effector_task.set_target_from_mocap(data, 0)

            # Compute velocity, integrate into position targets and set control signal.
            velocity = mink.solve_ik(
                configuration, tasks, limits, dt, prev_sol=velocity
            )
            data.ctrl = configuration.integrate(velocity, dt)
            mujoco.mj_step(model, data, sim_steps_per_control_steps)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
