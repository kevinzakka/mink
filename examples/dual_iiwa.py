"""Task adapted from https://github.com/stephane-caron/pink/pull/94."""

from dm_control import mjcf
import mujoco
import mujoco.viewer
import numpy as np
import mink
from pathlib import Path
from loop_rate_limiters import RateLimiter
from mink.utils import set_mocap_pose_from_site


_HERE = Path(__file__).parent
_XML = _HERE / "kuka_iiwa_14" / "iiwa14.xml"


def construct_model():
    root = mjcf.RootElement()
    root.statistic.meansize = 0.08
    getattr(root.visual, "global").azimuth = -120
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

    l_geom_names = [f"l_iiwa/geom_{i}" for i in range(30, 61)]
    r_geom_names = [f"r_iiwa/geom_{i}" for i in range(30, 61)]
    collision_pairs = [(l_geom_names, r_geom_names)]

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
    solver = "quadprog"

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

        set_mocap_pose_from_site(model, data, "l_target", "l_iiwa/attachment_site")
        set_mocap_pose_from_site(model, data, "r_target", "r_iiwa/attachment_site")

        rate = RateLimiter(frequency=60.0)
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
            left_ee_task.set_target_from_mocap(data, left_mid)
            right_ee_task.set_target_from_mocap(data, right_mid)

            vel = mink.solve_ik(configuration, tasks, limits, rate.dt, solver, 1e-2)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            viewer.sync()
            rate.sleep()
            t += rate.dt
