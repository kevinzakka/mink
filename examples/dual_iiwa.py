"""Task adapted from https://github.com/stephane-caron/pink/pull/94."""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from dm_control import mjcf
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "kuka_iiwa_14" / "iiwa14.xml"


def construct_model():
    root = mjcf.RootElement()
    root.statistic.meansize = 0.08
    root.statistic.extent = 1.0
    root.statistic.center = (0, 0, 0.5)
    getattr(root.visual, "global").azimuth = -180
    getattr(root.visual, "global").elevation = -20

    root.worldbody.add("light", pos="0 0 1.5", directional="true")

    left_site = root.worldbody.add(
        "site", name="l_attachment_site", pos=[0, 0.2, 0], group=5
    )
    right_site = root.worldbody.add(
        "site", name="r_attachment_site", pos=[0, -0.2, 0], group=5
    )

    left_iiwa = mjcf.from_path(_XML.as_posix())
    left_iiwa.model = "l_iiwa"
    left_iiwa.find("key", "home").remove()
    left_site.attach(left_iiwa)
    for i, g in enumerate(left_iiwa.worldbody.find_all("geom")):
        g.name = f"geom_{i}"

    right_iiwa = mjcf.from_path(_XML.as_posix())
    right_iiwa.model = "r_iiwa"
    right_iiwa.find("key", "home").remove()
    right_site.attach(right_iiwa)
    for i, g in enumerate(right_iiwa.worldbody.find_all("geom")):
        g.name = f"geom_{i}"

    body = root.worldbody.add("body", name="l_target", mocap=True)
    body.add(
        "geom",
        type="box",
        size=".05 .05 .05",
        contype="0",
        conaffinity="0",
        rgba=".3 .6 .3 .5",
    )

    body = root.worldbody.add("body", name="r_target", mocap=True)
    body.add(
        "geom",
        type="box",
        size=".05 .05 .05",
        contype="0",
        conaffinity="0",
        rgba=".3 .3 .6 .5",
    )

    return mujoco.MjModel.from_xml_string(root.to_xml_string(), root.get_assets())


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
                B + (A - B + 0.2 * np.array([0, 0, -np.sin(mu * np.pi) ** 2])) * mu
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
