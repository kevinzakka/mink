import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
from robot_descriptions import ur5e_mj_description

import mink
from mink.utils import pose_from_mocap, set_mocap_pose_from_site

from . import utils


def get_model() -> mujoco.MjModel:
    mjcf = utils.Mjcf.from_xml_path(ur5e_mj_description.MJCF_PATH)
    mjcf.add_checkered_plane()

    # Add obstacle.
    body = mjcf.add("body", name="wall", pos=(0.5, 0, 0.1))
    mjcf.add(
        "geom",
        parent=body,
        name="wall",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.1, 0.1, 0.1),
        contype=0,
        conaffinity=0,
    )

    # Add mocap target.
    body = mjcf.add("body", name="target", mocap=True)
    mjcf.add(
        "geom",
        parent=body,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.05,) * 3,
        contype=0,
        conaffinity=0,
        rgba=(0.6, 0.3, 0.3, 0.2),
    )

    return mjcf.compile()


if __name__ == "__main__":
    model = get_model()

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
    # collision_pairs = [
    #     (["wrist_3_link"], ["floor", "wall"]),
    # ]

    limits = [
        mink.ConfigurationLimit(model=model),
        # mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs),
    ]

    max_velocities = {
        "shoulder_pan_joint": np.pi,
        "shoulder_lift_joint": np.pi,
        "elbow_joint": np.pi,
        "wrist_1_joint": np.pi,
        "wrist_2_joint": np.pi,
        "wrist_3_joint": np.pi,
    }
    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)

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
            end_effector_task.set_target(pose_from_mocap(model, data, mid))

            # Compute velocity and integrate into the next configuration.
            vel = mink.solve_ik(configuration, tasks, limits, rate.dt, solver, 1e-3)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Note the below are optional: they are used to visualize the output of the
            # fromto sensor which is used by the collision avoidance constraint.
            # mujoco.mj_fwdPosition(model, data)
            # mujoco.mj_sensorPos(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
