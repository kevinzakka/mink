"""Stretch shuttling its base between stations while its gripper stays put.

The base target steps from one station to the other, but a free-joint velocity limit
bounds the floating base, so it eases across at a capped speed rather than snapping
there in a single solver step. ``VelocityLimit`` skips free joints, so the base
carries its own. Meanwhile the telescoping arm extends and retracts to hold the
gripper pinned in the world, the way you keep a hand planted on the table while
scooting your chair sideways.

Watch the base shuttle between stations::

    uv run mjpython examples/mobile_stretch.py
"""

from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "hello_robot_stretch_3" / "scene_33.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)

    tasks = [
        base_task := mink.FrameTask(
            frame_name="base_link",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
        ),
        fingertip_task := mink.FrameTask(
            frame_name="link_grasp_center",
            frame_type="site",
            position_cost=20.0,
            orientation_cost=20.0,
        ),
    ]

    # VelocityLimit skips free joints, so bound the floating base separately.
    limits = [
        mink.ConfigurationLimit(model),
        mink.FreeJointVelocityLimit(
            model,
            max_linear_velocity=0.2,
            max_angular_velocity=1.0,
        ),
    ]

    model = configuration.model
    data = configuration.data
    solver = "daqp"
    station_offset = 0.2  # [m]; lateral distance of each station from the start.
    station_dwell = 3.0  # [s]; time at a station before the target jumps.
    base_mid = model.body("base_target").mocapid[0]
    ee_mid = model.body("EE_target").mocapid[0]

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Pre-extend the telescoping arm to mid-range so the base can step either
        # way without the arm bottoming out against itself.
        configuration.update_from_keyframe("home")
        q = configuration.q.copy()
        for i in range(4):
            q[model.jnt_qposadr[model.joint(f"joint_arm_l{i}").id]] = 0.065
        configuration.update(q)
        base_task.set_target_from_configuration(configuration)
        assert base_task.transform_target_to_world is not None

        # Stations sit along the base lateral axis, the direction the arm extends.
        base_position = base_task.transform_target_to_world.translation()
        base_orientation = base_task.transform_target_to_world.rotation()
        base_lateral_axis = base_orientation.as_matrix()[:, 1]

        # Pin the gripper to its current world pose for the whole run.
        transform_gripper_pinned = configuration.get_transform_frame_to_world(
            "link_grasp_center", "site"
        )
        fingertip_task.set_target(transform_gripper_pinned)
        data.mocap_pos[ee_mid] = transform_gripper_pinned.translation()
        data.mocap_quat[ee_mid] = transform_gripper_pinned.rotation().wxyz

        rate = RateLimiter(frequency=100.0, warn=False)
        dt = rate.period
        t = 0.0
        while viewer.is_running():
            # Jump the base target between the two stations; the velocity limit
            # turns each jump into a bounded glide.
            offset = (
                station_offset if int(t / station_dwell) % 2 == 0 else -station_offset
            )
            transform_base_target = mink.SE3.from_rotation_and_translation(
                base_orientation,
                base_position + offset * base_lateral_axis,
            )
            base_task.set_target(transform_base_target)

            data.mocap_pos[base_mid] = transform_base_target.translation()
            data.mocap_quat[base_mid] = transform_base_target.rotation().wxyz

            # Compute velocity and integrate into the next configuration.
            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, damping=1e-3, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
            t += dt
