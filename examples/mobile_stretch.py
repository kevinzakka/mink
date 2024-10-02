from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "hello_robot_stretch_3" / "scene.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)

    tasks = [
        base_task := mink.FrameTask(
            frame_name="base_link",
            frame_type="body",
            position_cost=0.1,
            orientation_cost=1.0,
        ),
        fingertip_task := mink.FrameTask(
            frame_name="link_grasp_center",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1e-4,
        ),
    ]

    model = configuration.model
    data = configuration.data
    solver = "quadprog"
    circle_radius = 0.5
    mid = model.body("base_target").mocapid[0]

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("home")
        base_task.set_target_from_configuration(configuration)
        assert base_task.transform_target_to_world is not None

        transform_fingertip_target_to_world = (
            configuration.get_transform_frame_to_world("link_grasp_center", "site")
        )
        center_translation = transform_fingertip_target_to_world.translation()[:2]
        fingertip_task.set_target(transform_fingertip_target_to_world)
        mink.move_mocap_to_frame(model, data, "EE_target", "link_grasp_center", "site")

        rate = RateLimiter(frequency=100.0, warn=False)
        dt = rate.period
        t = 0.0
        while viewer.is_running():
            # Update task targets
            u = np.array([np.cos(t / 2), np.sin(t / 2)])
            T = base_task.transform_target_to_world.copy()
            translation = T.translation()
            translation[:2] = center_translation + circle_radius * u
            data.mocap_pos[mid] = translation
            data.mocap_quat[mid] = mink.SO3.from_rpy_radians(
                0.0, 0.0, 0.5 * np.pi * t
            ).wxyz
            base_task.set_target(mink.SE3.from_mocap_id(data, mid))

            # Compute velocity and integrate into the next configuration.
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
            t += dt
