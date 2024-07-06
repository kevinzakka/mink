from dm_control import mjcf
import mujoco
import mujoco.viewer
import numpy as np
import mink
from pathlib import Path
from loop_rate_limiters import RateLimiter
from mink.utils import set_mocap_pose_from_site


_HERE = Path(__file__).parent
_XML = _HERE / "universal_robots_ur5e" / "ur5e.xml"


def construct_model():
    root = mjcf.RootElement()
    root.statistic.meansize = 0.08
    getattr(root.visual, "global").azimuth = -120
    getattr(root.visual, "global").elevation = -20

    root.worldbody.add("light", pos="0 0 1.5", directional="true")

    base = root.worldbody.add("body", name="base")
    width, height, depth = 0.8, 0.4, 0.2
    base.add(
        "geom",
        type="box",
        size=[width, height, depth],
        density=1e-3,
        rgba=".9 .8 .6 1",
    )
    base.pos = [-0.0 * width, -0.0 * height, -0.5 * depth]
    base.add("freejoint")
    base.add("site", name="base", pos=[0, 0, depth], group=1)
    left_site = base.add(
        "site", name="left_attachment_site", pos=[0.3, 0, depth], group=5
    )
    right_site = base.add(
        "site",
        name="right_attachment_site",
        pos=[-0.3, 0, depth],
        group=5,
    )

    left_ur5e = mjcf.from_path(_XML.as_posix())
    left_ur5e.model = "left_ur5e"
    left_ur5e.find("key", "home").remove()
    left_site.attach(left_ur5e)

    right_ur5e = mjcf.from_path(_XML.as_posix())
    right_ur5e.model = "right_ur5e"
    right_ur5e.find("key", "home").remove()
    right_site.attach(right_ur5e)

    body = root.worldbody.add("body", name="base_target", mocap=True)
    body.add(
        "geom",
        type="box",
        size=".05 .05 .05",
        contype="0",
        conaffinity="0",
        rgba=".6 .3 .3 .5",
    )

    body = root.worldbody.add("body", name="left_target", mocap=True)
    body.add(
        "geom",
        type="box",
        size=".05 .05 .05",
        contype="0",
        conaffinity="0",
        rgba=".3 .6 .3 .5",
    )

    body = root.worldbody.add("body", name="right_target", mocap=True)
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

    base_task = mink.FrameTask(
        frame_name="base",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    left_ee_task = mink.FrameTask(
        frame_name="left_ur5e/attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    right_ee_task = mink.FrameTask(
        frame_name="right_ur5e/attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    tasks = [base_task, left_ee_task, right_ee_task]
    for task in tasks:
        task.set_target_from_configuration(configuration)

    limits = [
        mink.ConfigurationLimit(model=model),
    ]

    model = configuration.model
    data = configuration.data
    velocity = None

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        base_mocap_id = model.body("base_target").mocapid[0]
        set_mocap_pose_from_site(model, data, "base_target", "base")

        left_mocap_id = model.body("left_target").mocapid[0]
        set_mocap_pose_from_site(
            model, data, "left_target", "left_ur5e/attachment_site"
        )

        right_mocap_id = model.body("right_target").mocapid[0]
        set_mocap_pose_from_site(
            model, data, "right_target", "right_ur5e/attachment_site"
        )

        rate = RateLimiter(frequency=200.0)
        dt = rate.period
        t = 0.0
        while viewer.is_running():
            data.mocap_pos[base_mocap_id][2] = 0.3 * np.sin(2.0 * t)
            base_task.set_target_from_mocap(data, base_mocap_id)

            data.mocap_pos[left_mocap_id][1] = 0.5 + 0.2 * np.sin(2.0 * t)
            data.mocap_pos[left_mocap_id][2] = 0.2
            left_ee_task.set_target_from_mocap(data, left_mocap_id)

            data.mocap_pos[right_mocap_id][1] = 0.5 + 0.2 * np.sin(2.0 * t)
            data.mocap_pos[right_mocap_id][2] = 0.2
            right_ee_task.set_target_from_mocap(data, right_mocap_id)

            velocity = mink.solve_ik(configuration, tasks, limits, dt, 1e-2, velocity)
            configuration.integrate_inplace(velocity, dt)

            viewer.sync()
            rate.sleep()
            t += dt
