import mujoco
import mujoco.viewer
import mink
import numpy as np
from loop_rate_limiters import RateLimiter
from pathlib import Path

_HERE = Path(__file__).parent
_MODEL_PATH = _HERE / "franka_emika_panda" / "dual_panda_scene.xml"


def initialize_model():
    model = mujoco.MjModel.from_xml_path(_MODEL_PATH.as_posix())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # Force update after stepping
    configuration = mink.Configuration(model)
    return model, data, configuration


def ik_task_constraints(model):
    left_ee_task = mink.FrameTask(
        frame_name="attachment_site_left",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    right_ee_task = mink.FrameTask(
        frame_name="attachment_site_right",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    tasks = [left_ee_task, right_ee_task, posture_task]
    solver = "osqp"
    pos_threshold, ori_threshold = 0.008, 0.008
    max_iters = 20  # IK iterations per cycle
    rate = RateLimiter(frequency=40.0, warn=False)
    return (
        left_ee_task,
        right_ee_task,
        posture_task,
        solver,
        tasks,
        pos_threshold,
        ori_threshold,
        max_iters,
        rate,
    )


def define_waypoints(
    data, site_ref_left, site_ref_right, target_quat_left, target_quat_right
):
    """Define example waypoints for both arms."""
    # Positions
    start_left_pos = data.site_xpos[site_ref_left].copy()
    start_left_pos[0] -= 0.05
    start_right_pos = data.site_xpos[site_ref_right].copy()
    start_right_pos[0] += 0.05

    middle_left_pos = start_left_pos.copy()
    middle_left_pos[0] += 0.07
    middle_right_pos = start_right_pos.copy()
    middle_right_pos[0] -= 0.07

    # Waypoints
    left_waypoints = [
        (start_left_pos, target_quat_left.copy()),
        (middle_left_pos, target_quat_left.copy()),
    ]
    right_waypoints = [
        (start_right_pos, target_quat_right.copy()),
        (middle_right_pos, target_quat_right.copy()),
    ]

    # Lift positions
    lifted_left_pos = middle_left_pos.copy()
    lifted_left_pos[2] += 0.36
    lifted_right_pos = middle_right_pos.copy()
    lifted_right_pos[2] += 0.36

    return left_waypoints, right_waypoints, lifted_left_pos, lifted_right_pos


def check_reached(
    meas_left_pos,
    tgt_left_pos,
    meas_left_quat,
    tgt_left_quat,
    meas_right_pos,
    tgt_right_pos,
    meas_right_quat,
    tgt_right_quat,
    pos_threshold,
    ori_threshold,
):
    """Returns True if both arms have reached position/orientation targets."""
    err_pos_left = np.linalg.norm(meas_left_pos - tgt_left_pos)
    err_ori_left = quaternion_error(meas_left_quat, tgt_left_quat)
    err_pos_right = np.linalg.norm(meas_right_pos - tgt_right_pos)
    err_ori_right = quaternion_error(meas_right_quat, tgt_right_quat)

    return (
        err_pos_left <= pos_threshold
        and err_ori_left <= ori_threshold
        and err_pos_right <= pos_threshold
        and err_ori_right <= ori_threshold
    )


def initialize_mocap_targets(model, data):
    """Initialize the mocap targets to match the initial end-effector sites."""
    site_ref_left = model.site("site_left").id
    site_ref_right = model.site("site_right").id
    body_left_mocap_id = model.body("target_left").mocapid
    body_right_mocap_id = model.body("target_right").mocapid

    # Position
    data.mocap_pos[body_left_mocap_id] = data.site_xpos[site_ref_left].copy()
    data.mocap_pos[body_right_mocap_id] = data.site_xpos[site_ref_right].copy()

    # Orientation
    q_left = np.zeros(4)
    q_right = np.zeros(4)
    mujoco.mju_mat2Quat(q_left, data.site_xmat[site_ref_left])
    mujoco.mju_mat2Quat(q_right, data.site_xmat[site_ref_right])
    data.mocap_quat[body_left_mocap_id] = q_left
    data.mocap_quat[body_right_mocap_id] = q_right

    return q_left, q_right


def update_mocap_targets(model, data, left_pos, left_quat, right_pos, right_quat):
    """Set the position/orientation of both targets in the MjData."""
    data.mocap_pos[model.body("target_left").mocapid] = left_pos
    data.mocap_quat[model.body("target_left").mocapid] = left_quat
    data.mocap_pos[model.body("target_right").mocapid] = right_pos
    data.mocap_quat[model.body("target_right").mocapid] = right_quat


def quaternion_error(q_current, q_target):
    """Returns the minimum of (||q_cur - q_tar||, ||q_cur + q_tar||)."""
    err1 = np.linalg.norm(q_current - q_target)
    err2 = np.linalg.norm(q_current + q_target)
    return min(err1, err2)


def ik_iteration(
    configuration,
    tasks,
    dt,
    solver,
    data,
    model,
    site_left_id,
    site_right_id,
    left_target_pos,
    left_target_quat,
    right_target_pos,
    right_target_quat,
    pos_threshold,
    ori_threshold,
    gripper_val=255,
):
    """
    Performs one IK iteration and checks if the targets are reached.
    Returns True if both arms reached the target, otherwise False.
    """
    # Solve IK and integrate
    vel = mink.solve_ik(configuration, tasks, dt, solver, 5e-3)
    configuration.integrate_inplace(vel, dt)

    # Update controls
    data.ctrl[0:7] = configuration.q[0:7]
    data.ctrl[8:15] = configuration.q[9:16]
    data.ctrl[7] = gripper_val
    data.ctrl[15] = gripper_val

    # Step the simulation
    mujoco.mj_step(model, data)

    # Measure
    measured_left_pos = data.site_xpos[site_left_id]
    measured_right_pos = data.site_xpos[site_right_id]
    measured_left_quat = np.zeros(4)
    measured_right_quat = np.zeros(4)
    mujoco.mju_mat2Quat(measured_left_quat, data.site_xmat[site_left_id])
    mujoco.mju_mat2Quat(measured_right_quat, data.site_xmat[site_right_id])

    # Check if targets are reached
    return check_reached(
        measured_left_pos,
        left_target_pos,
        measured_left_quat,
        left_target_quat,
        measured_right_pos,
        right_target_pos,
        measured_right_quat,
        right_target_quat,
        pos_threshold,
        ori_threshold,
    )


def move_gripper(model, data, viewer, rate, steps=100, gripper_val=0):
    """
    Closes or opens the gripper by setting ctrl to the specified value,
    stepping the simulation, and syncing the viewer.
    """
    for _ in range(steps):
        data.ctrl[7], data.ctrl[15] = gripper_val, gripper_val
        mujoco.mj_step(model, data)  # Use model and data from parameters
        viewer.sync()
        rate.sleep()


def run_waypoints(
    model,
    data,
    configuration,
    viewer,
    left_waypoints,
    right_waypoints,
    left_ee_task,
    right_ee_task,
    tasks,
    solver,
    site_left_id,
    site_right_id,
    site_ref_left,
    site_ref_right,
    pos_threshold,
    ori_threshold,
    max_iters,
    rate,
):
    """Plans and executes each waypoint until reached."""
    for wp_index, (left_wp, right_wp) in enumerate(
        zip(left_waypoints, right_waypoints)
    ):
        print(f"\nPlanning to waypoint {wp_index + 1}:")

        # Unpack
        current_left_pos, current_left_quat = left_wp
        current_right_pos, current_right_quat = right_wp

        reached = False
        while not reached and viewer.is_running():
            # Update mocap and tasks
            update_mocap_targets(
                model,
                data,
                current_left_pos,
                current_left_quat,
                current_right_pos,
                current_right_quat,
            )
            T_wt_left = mink.SE3.from_mocap_name(model, data, "target_left")
            T_wt_right = mink.SE3.from_mocap_name(model, data, "target_right")
            left_ee_task.set_target(T_wt_left)
            right_ee_task.set_target(T_wt_right)

            # IK iterations
            for _ in range(max_iters):
                reached = ik_iteration(
                    configuration,
                    tasks,
                    rate.dt,
                    solver,
                    data,
                    model,
                    site_left_id,
                    site_right_id,
                    current_left_pos,
                    current_left_quat,
                    current_right_pos,
                    current_right_quat,
                    pos_threshold,
                    ori_threshold,
                    gripper_val=255,
                )
                viewer.sync()
                rate.sleep()
                if reached:
                    break

            # If still not reached, do one more step for stability
            if not reached:
                mujoco.mj_step(model, data)
                viewer.sync()
                rate.sleep()

        print(f"Waypoint {wp_index + 1} reached.\n")


def main():
    model, data, configuration = initialize_model()
    # Initialize tasks, solver, thresholds, etc.
    (
        left_ee_task,
        right_ee_task,
        posture_task,
        solver,
        tasks,
        pos_threshold,
        ori_threshold,
        max_iters,
        rate,
    ) = ik_task_constraints(model)

    # Grab site IDs
    site_left_id = model.site("attachment_site_left").id
    site_right_id = model.site("attachment_site_right").id
    site_ref_left = model.site("site_left").id
    site_ref_right = model.site("site_right").id

    # Initialize mocap targets
    tgt_quat_left, tgt_quat_right = initialize_mocap_targets(model, data)

    # Create example waypoints
    left_waypoints, right_waypoints, lifted_left_pos, lifted_right_pos = (
        define_waypoints(
            data, site_ref_left, site_ref_right, tgt_quat_left, tgt_quat_right
        )
    )

    # Launch viewer
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home1").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # Execute waypoints
        run_waypoints(
            model,
            data,
            configuration,
            viewer,
            left_waypoints,
            right_waypoints,
            left_ee_task,
            right_ee_task,
            tasks,
            solver,
            site_left_id,
            site_right_id,
            site_ref_left,
            site_ref_right,
            pos_threshold,
            ori_threshold,
            max_iters,
            rate,
        )

        # Close gripper
        print("All waypoints reached. Closing gripper...")
        move_gripper(model, data, viewer, rate, steps=100, gripper_val=0)
        print("Gripper closed.")

        # Lift object
        data.mocap_pos[model.body("target_left").mocapid] = lifted_left_pos
        data.mocap_pos[model.body("target_right").mocapid] = lifted_right_pos
        T_wt_left = mink.SE3.from_mocap_name(model, data, "target_left")
        T_wt_right = mink.SE3.from_mocap_name(model, data, "target_right")
        left_ee_task.set_target(T_wt_left)
        right_ee_task.set_target(T_wt_right)

        # Final IK steps to lift
        for _ in range(max_iters):
            ik_iteration(
                configuration,
                tasks,
                rate.dt,
                solver,
                data,
                model,
                site_left_id,
                site_right_id,
                lifted_left_pos,
                tgt_quat_left,
                lifted_right_pos,
                tgt_quat_right,
                pos_threshold,
                ori_threshold,
                gripper_val=0,
            )
            viewer.sync()
            rate.sleep()

        # Keep viewer open
        while viewer.is_running():
            data.ctrl[7], data.ctrl[15] = 0, 0
            mujoco.mj_step(model, data)
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()
