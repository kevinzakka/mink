from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "the_robot_studio_so101" / "scene.xml"

# IK parameters
SOLVER = "daqp"
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4
MAX_ITERS = 20

# SO101 end effector details
END_EFFECTOR = "gripperframe"
END_EFFECTOR_TYPE = "site"


def converge_ik(configuration, tasks, dt, solver, pos_threshold, ori_threshold, max_iters):
    """Runs up to 'max_iters' of IK steps. Returns True if position and orientation
    are below thresholds, otherwise False."""
    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, tasks.values(), dt, solver, damping=1e-3)
        configuration.integrate_inplace(vel, dt)

        # Only checking the first FrameTask here (end_effector_task).
        # If you want to check multiple tasks, sum or combine their errors.
        err = tasks["eef"].compute_error(configuration)
        pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
        ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold

        if pos_achieved and ori_achieved:
            return True
    return False


def main():
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    configuration = mink.Configuration(model)

    end_effector_task = mink.FrameTask(
        frame_name=END_EFFECTOR, frame_type=END_EFFECTOR_TYPE, position_cost=1.0, orientation_cost=1.0, lm_damping=1.0
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    tasks = {"eef": end_effector_task, "posture": posture_task}

    model.vis.scale.framewidth = 0.01
    model.vis.scale.framelength = 0.5
    # Initialize viewer in passive mode
    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        with viewer.lock():
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_GEOM

        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        mink.move_mocap_to_frame(model, data, "target", END_EFFECTOR, END_EFFECTOR_TYPE)
        initial_target_position = data.mocap_pos[0].copy()

        # Circular trajectory parameters.
        amp = 0.05
        freq = 0.2
        # Gripper rotation parameters
        angle_amp = 30  # degree
        angle_freq = 0.5

        # We'll track time ourselves for a smoother trajectory.
        local_time = 0.0
        rate = RateLimiter(frequency=200.0, warn=False)

        while viewer.is_running():
            dt = rate.dt
            local_time += dt

            # Circular offset.
            offset = np.array(
                [
                    amp * np.cos(2 * np.pi * freq * local_time),
                    0.0,
                    amp * np.sin(2 * np.pi * freq * local_time),
                ]
            )
            data.mocap_pos[0] = initial_target_position + offset
            # Gripper rotation around target x axis
            quaternion = np.array(
                [
                    np.cos(np.radians(angle_amp * np.cos(2 * np.pi * angle_freq * local_time)) / 2),
                    np.sin(np.radians(angle_amp * np.cos(2 * np.pi * angle_freq * local_time)) / 2),
                    0,
                    0,
                ]
            )
            data.mocap_quat[0] = quaternion

            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            converge_ik(configuration, tasks, dt, SOLVER, POS_THRESHOLD, ORI_THRESHOLD, MAX_ITERS)

            data.ctrl = configuration.q[:6]
            mujoco.mj_step(model, data)

            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()
