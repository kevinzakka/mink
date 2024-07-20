from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink
from mink.utils import (
    set_mocap_pose_from_body,
    set_mocap_pose_from_geom,
    set_mocap_pose_from_site,
)

_HERE = Path(__file__).parent
_XML = _HERE / "boston_dynamics_spot" / "scene.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)

    feet = ["FL", "FR", "HR", "HL"]

    base_task = mink.FrameTask(
        frame_name="body",
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    posture_task = mink.PostureTask(cost=1e-5)

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="geom",
            position_cost=1.0,
            orientation_cost=0.0,
        )
        feet_tasks.append(task)

    eef_task = mink.FrameTask(
        frame_name="EE",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    tasks = [base_task, posture_task, *feet_tasks, eef_task]

    # # Get all geom IDs belonging to the arm.
    # arm_body_ids: list[int] = []
    # for bid in range(model.nbody):
    #     if "arm" in model.body(bid).name:
    #         arm_body_ids.append(bid)
    # arm_geom_ids: list[int] = []
    # for gid in range(model.ngeom):
    #     if model.geom_bodyid[gid] in arm_body_ids:
    #         arm_geom_ids.append(gid)
    # body_ids: list[int] = []
    # for bid in range(model.nbody):
    #     if model.body_rootid[bid] == model.body("body").id and bid not in arm_body_ids:
    #         body_ids.append(bid)
    # quad_geom_ids: list[int] = []
    # for gid in range(model.ngeom):
    #     if model.geom_bodyid[gid] in body_ids:
    #         quad_geom_ids.append(gid)
    # collision_pairs = [(arm_geom_ids, quad_geom_ids)]

    limits = [
        mink.ConfigurationLimit(model=model),
        # mink.CollisionAvoidanceLimit(
        #     model=model,
        #     geom_pairs=collision_pairs,
        #     minimum_distance_from_collisions=0.01,
        #     collision_detection_distance=0.5,
        #     as_id=True,
        # ),
    ]

    base_mid = model.body("body_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
    eef_mid = model.body("EE_target").mocapid[0]

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
            set_mocap_pose_from_geom(model, data, f"{foot}_target", foot)
        set_mocap_pose_from_body(model, data, "body_target", "body")
        set_mocap_pose_from_site(model, data, "EE_target", "EE")

        rate = RateLimiter(frequency=500.0)
        while viewer.is_running():
            # Update task targets.
            base_task.set_target(mink.SE3.from_mocap(data, base_mid))
            for i, task in enumerate(feet_tasks):
                task.set_target(mink.SE3.from_mocap(data, feet_mid[i]))
            eef_task.set_target(mink.SE3.from_mocap(data, eef_mid))

            # Compute velocity, integrate and set control signal.
            vel = mink.solve_ik(configuration, tasks, limits, rate.dt, solver, 1e-5)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
