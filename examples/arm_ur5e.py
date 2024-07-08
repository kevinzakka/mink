import mujoco
import mujoco.viewer
import numpy as np
import mink
from pathlib import Path
from loop_rate_limiters import RateLimiter
from mink.utils import set_mocap_pose_from_site

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

    # Enable collision avoidance between the following geoms:
    collision_pairs = [
        (["wrist_3_link"], ["floor", "wall"]),
    ]

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, np.full_like(configuration.q, np.pi)),
        mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs),
    ]

    mid = model.body("target").mocapid[0]
    model = configuration.model
    data = configuration.data
    solver = "quadprog"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("home")

        # Initialize the mocap target at the end-effector site.
        set_mocap_pose_from_site(model, data, "target", "attachment_site")

        rate = RateLimiter(frequency=500.0)
        while viewer.is_running():
            # Update task target.
            end_effector_task.set_target_from_mocap(data, mid)

            # Compute velocity and integrate into the next configuration.
            vel = mink.solve_ik(configuration, tasks, limits, rate.dt, solver, 1e-3)
            configuration.integrate_inplace(vel, rate.dt)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
