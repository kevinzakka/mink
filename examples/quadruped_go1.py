import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
from robot_descriptions import go1_mj_description

import mink
from mink.utils import (
    pose_from_mocap,
    set_mocap_pose_from_body,
    set_mocap_pose_from_site,
)

from . import utils


def get_model() -> mujoco.MjModel:
    mjcf = utils.Mjcf.from_xml_path(go1_mj_description.MJCF_PATH)
    mjcf.add_checkered_plane()

    body = mjcf.add("body", name="trunk_target", mocap=True)
    mjcf.add(
        "geom",
        parent=body,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.08,) * 3,
        contype=0,
        conaffinity=0,
        rgba=(0.6, 0.3, 0.3, 0.2),
    )

    for feet in ["FL", "FR", "RR", "RL"]:
        body = mjcf.add("body", name=f"{feet}_target", mocap=True)
        mjcf.add(
            "geom",
            parent=body,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.04,) * 3,
            contype=0,
            conaffinity=0,
            rgba=(0.6, 0.3, 0.3, 0.2),
        )

    return mjcf.compile()


if __name__ == "__main__":
    model = get_model()

    configuration = mink.Configuration(model)

    feet = ["FL", "FR", "RR", "RL"]

    base_task = mink.FrameTask(
        frame_name="trunk",
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    posture_task = mink.PostureTask(model, cost=1e-5)

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

    limits = [
        mink.ConfigurationLimit(model=model),
    ]

    model = configuration.model
    data = configuration.data
    solver = "quadprog"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("home")
        posture_task.set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective sites.
        for foot in feet:
            set_mocap_pose_from_site(model, data, f"{foot}_target", foot)
        set_mocap_pose_from_body(model, data, "trunk_target", "trunk")

        rate = RateLimiter(frequency=500.0)
        while viewer.is_running():
            # Update task targets.
            base_task.set_target(pose_from_mocap(model, data, "trunk_target"))
            for i, task in enumerate(feet_tasks):
                task.set_target(pose_from_mocap(model, data, f"{feet[i]}_target"))

            # Compute velocity, integrate and set control signal.
            vel = mink.solve_ik(configuration, tasks, limits, rate.dt, solver, 1e-5)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
