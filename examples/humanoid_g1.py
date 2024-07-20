import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
from robot_descriptions import g1_mj_description

import mink
from mink.utils import set_mocap_pose_from_site

from . import utils


def get_model() -> mujoco.MjModel:
    mjcf = utils.Mjcf.from_xml_path(g1_mj_description.MJCF_PATH)
    mjcf.add_checkered_plane()

    # Add palm sites.
    # TODO(kevin): Remove once added to menagerie.
    body = mjcf.spec.find_body("left_zero_link")
    site = body.add_site()
    site.name = "left_palm"
    body = mjcf.spec.find_body("right_zero_link")
    site = body.add_site()
    site.name = "right_palm"

    body = mjcf.add("body", name="com_target", mocap=True)
    mjcf.add(
        "geom",
        parent=body,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.08,) * 3,
        contype=0,
        conaffinity=0,
        rgba=(0.6, 0.3, 0.3, 0.2),
    )

    for feet in ["left_foot", "right_foot"]:
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

    for feet in ["left_palm", "right_palm"]:
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

    feet = ["right_foot", "left_foot"]
    hands = ["right_palm", "left_palm"]

    tasks = [
        pelvis_orientation_task := mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=10.0,
        ),
        posture_task := mink.PostureTask(model, cost=1e-1),
        com_task := mink.ComTask(cost=10.0),
    ]

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        )
        feet_tasks.append(task)
    tasks.extend(feet_tasks)

    hand_tasks = []
    for hand in hands:
        task = mink.FrameTask(
            frame_name=hand,
            frame_type="site",
            position_cost=4.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        hand_tasks.append(task)
    tasks.extend(hand_tasks)

    limits = [
        mink.ConfigurationLimit(model=model),
    ]

    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
    hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in hands]

    model = configuration.model
    data = configuration.data
    solver = "quadprog"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("stand")
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective sites.
        for hand, foot in zip(hands, feet):
            set_mocap_pose_from_site(model, data, f"{foot}_target", foot)
            set_mocap_pose_from_site(model, data, f"{hand}_target", hand)
        data.mocap_pos[com_mid] = data.subtree_com[1]

        rate = RateLimiter(frequency=500.0)
        while viewer.is_running():
            # Update task targets.
            com_task.set_target(data.mocap_pos[com_mid])
            for i, (hand_task, foot_task) in enumerate(zip(hand_tasks, feet_tasks)):
                foot_task.set_target(mink.SE3.from_mocap(data, feet_mid[i]))
                hand_task.set_target(mink.SE3.from_mocap(data, hands_mid[i]))

            vel = mink.solve_ik(
                configuration, tasks, limits, rate.dt, solver, 1e-1, safety_break=False
            )
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
