import mujoco
import mujoco.viewer
import mink
from pathlib import Path
from loop_rate_limiters import RateLimiter
from mink.utils import set_mocap_pose_from_site

_HERE = Path(__file__).parent
_XML = _HERE / "unitree_g1" / "scene.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    feet = ["right_foot", "left_foot"]
    hands = ["right_palm", "left_palm"]

    #
    # Limits.
    #

    limits = [
        mink.ConfigurationLimit(model=model),
    ]

    #
    # Tasks.
    #

    tasks = [
        pelvis_orientation_task := mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=10.0,
        ),
        posture_task := mink.PostureTask(cost=1e-1),
        com_task := mink.ComTask(cost=200.0),
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

    configuration = mink.Configuration(model)
    model = configuration.model
    data = configuration.data
    velocity = None

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Initialize to the home keyframe.
        mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
        configuration.update(lights=True)
        for task in tasks:
            task.set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective targetees.
        for foot in feet:
            set_mocap_pose_from_site(model, data, f"{foot}_target", foot)
        for hand in hands:
            set_mocap_pose_from_site(model, data, f"{hand}_target", hand)

        com_mocap_id = model.body("com_target").mocapid[0]
        data.mocap_pos[com_mocap_id] = data.subtree_com[1]

        # Initialize the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        rate = RateLimiter(frequency=500.0)
        dt = rate.period
        while viewer.is_running():
            # Update task targets.
            for i, task in enumerate(hand_tasks):
                mocap_id = model.body(f"{hands[i]}_target").mocapid[0]
                task.set_target_from_mocap(data, mocap_id)
            for i, task in enumerate(feet_tasks):
                mocap_id = model.body(f"{feet[i]}_target").mocapid[0]
                task.set_target_from_mocap(data, mocap_id)
            com_task.set_target_from_mocap(data, com_mocap_id)

            velocity = mink.solve_ik(configuration, tasks, limits, dt, 1e-1, velocity)
            configuration.integrate_inplace(velocity, dt)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
