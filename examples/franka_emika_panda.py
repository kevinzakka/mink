from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "franka_emika_panda" / "mjx_scene.xml"

# IK parameters
SOLVER = "quadprog"
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4
MAX_ITERS = 20


def converge_ik(
    configuration, tasks, dt, solver, pos_threshold, ori_threshold, max_iters
):
    """
    Runs up to 'max_iters' of IK steps. Returns True if position and orientation
    are below thresholds, otherwise False.
    """
    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, tasks, dt, solver, 1e-3)
        configuration.integrate_inplace(vel, dt)

        # Only checking the first FrameTask here (end_effector_task).
        # If you want to check multiple tasks, sum or combine their errors.
        err = tasks[0].compute_error(configuration)
        pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
        ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold

        if pos_achieved and ori_achieved:
            return True
    return False


def main():
    # Load model & data
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Create a Mink configuration
    configuration = mink.Configuration(model)

    # Define tasks
    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    tasks = [end_effector_task, posture_task]

    # Initialize viewer in passive mode
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Reset simulation data to the 'home' keyframe
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # Move the mocap target to the end-effector's current pose
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
        initial_target_position = data.mocap_pos[0].copy()

        # Circular trajectory parameters
        amp = 0.10
        freq = 0.2

        # We'll track time ourselves for a smoother trajectory
        local_time = 0.0
        rate = RateLimiter(frequency=200.0, warn=False)

        while viewer.is_running():
            # Update our local time
            dt = rate.dt
            local_time += dt

            # Circular offset
            offset = np.array(
                [
                    amp * np.cos(2 * np.pi * freq * local_time),
                    amp * np.sin(2 * np.pi * freq * local_time),
                    0.0,
                ]
            )
            data.mocap_pos[0] = initial_target_position + offset

            # Update the end effector task target from the mocap body
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            # Attempt to converge IK
            converge_ik(
                configuration,
                tasks,
                dt,
                SOLVER,
                POS_THRESHOLD,
                ORI_THRESHOLD,
                MAX_ITERS,
            )

            # Set robot controls (first 8 dofs in your configuration)
            data.ctrl = configuration.q[:8]

            # Step simulation
            mujoco.mj_step(model, data)

            # Visualize at fixed FPS
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()
