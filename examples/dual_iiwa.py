"""Task adapted from https://github.com/stephane-caron/pink/pull/94."""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "kuka_iiwa_14" / "iiwa14.xml"


def construct_model() -> mujoco.MjModel:
    root = mujoco.MjSpec()
    root.stat.meansize = 0.08
    root.stat.extent = 1.0
    root.stat.center = (0, 0, 0.5)
    root.visual.global_.azimuth = -180
    root.visual.global_.elevation = -20

    root.worldbody.add_light(pos=(0, 0, 1.5), directional=True)

    left_site = root.worldbody.add_site(
        name="l_attachment_site", pos=[0, 0.2, 0], group=5
    )
    right_site = root.worldbody.add_site(
        name="r_attachment_site", pos=[0, -0.2, 0], group=5
    )

    left_iiwa = mujoco.MjSpec.from_file(_XML.as_posix())
    left_iiwa.modelname = "l_iiwa"
    left_iiwa.key("home").delete()
    for i in range(len(left_iiwa.geoms)):
        left_iiwa.geoms[i].name = f"geom_{i}"
    root.attach(left_iiwa, site=left_site, prefix="l_iiwa/")

    right_iiwa = mujoco.MjSpec.from_file(_XML.as_posix())
    right_iiwa.modelname = "r_iiwa"
    right_iiwa.key("home").delete()
    for i in range(len(right_iiwa.geoms)):
        right_iiwa.geoms[i].name = f"geom_{i}"
    root.attach(right_iiwa, site=right_site, prefix="r_iiwa/")

    body = root.worldbody.add_body(name="l_target", mocap=True)
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.05, 0.05, 0.05),
        contype=0,
        conaffinity=0,
        rgba=(0.3, 0.6, 0.3, 0.5),
    )

    body = root.worldbody.add_body(name="r_target", mocap=True)
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.05, 0.05, 0.05),
        contype=0,
        conaffinity=0,
        rgba=(0.3, 0.3, 0.6, 0.5),
    )

    return root.compile()


if __name__ == "__main__":
    model = construct_model()

    configuration = mink.Configuration(model)

    tasks = [
        left_ee_task := mink.FrameTask(
            frame_name="l_iiwa/attachment_site",
            frame_type="site",
            position_cost=2.0,
            orientation_cost=1.0,
        ),
        right_ee_task := mink.FrameTask(
            frame_name="r_iiwa/attachment_site",
            frame_type="site",
            position_cost=2.0,
            orientation_cost=1.0,
        ),
    ]

    collision_pairs = [
        (
            mink.get_subtree_geom_ids(model, model.body("l_iiwa/link5").id),
            mink.get_subtree_geom_ids(model, model.body("r_iiwa/link5").id),
        ),
    ]

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.CollisionAvoidanceLimit(
            model=model,
            geom_pairs=collision_pairs,
            minimum_distance_from_collisions=0.1,
            collision_detection_distance=0.2,
        ),
    ]

    left_mid = model.body("l_target").mocapid[0]
    right_mid = model.body("r_target").mocapid[0]
    model = configuration.model
    data = configuration.data
    solver = "osqp"

    l_y_des = np.array([0.392, -0.392, 0.6])
    r_y_des = np.array([0.392, 0.392, 0.6])
    A = l_y_des.copy()
    B = r_y_des.copy()
    l_dy_des = np.zeros(3)
    r_dy_des = np.zeros(3)

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        mink.move_mocap_to_frame(
            model, data, "l_target", "l_iiwa/attachment_site", "site"
        )
        mink.move_mocap_to_frame(
            model, data, "r_target", "r_iiwa/attachment_site", "site"
        )

        rate = RateLimiter(frequency=60.0, warn=False)
        t = 0.0
        while viewer.is_running():
            mu = (1 + np.cos(t)) / 2
            l_y_des[:] = (
                A + (B - A + 0.2 * np.array([0, 0, np.sin(mu * np.pi) ** 2])) * mu
            )
            r_y_des[:] = (
                B + (A - B + 0.2 * np.array([0, 0, -(np.sin(mu * np.pi) ** 2)])) * mu
            )
            data.mocap_pos[left_mid] = l_y_des
            data.mocap_pos[right_mid] = r_y_des

            # Update task targets.
            T_wt_left = mink.SE3.from_mocap_name(model, data, "l_target")
            left_ee_task.set_target(T_wt_left)
            T_wt_right = mink.SE3.from_mocap_name(model, data, "r_target")
            right_ee_task.set_target(T_wt_right)

            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, 1e-2, False, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            viewer.sync()
            rate.sleep()
            t += rate.dt
