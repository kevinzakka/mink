from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "aloha" / "scene.xml"

# Single arm joint names.
_JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]

# Single arm velocity limits, taken from:
# https://github.com/Interbotix/interbotix_ros_manipulators/blob/main/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/vx300s.urdf.xacro
_VELOCITY_LIMITS = {k: np.pi for k in _JOINT_NAMES}


def gravcomp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_ids: np.ndarray,
) -> np.ndarray:
    """Compute force to counteract gravity for the given bodies."""
    jac = np.empty((3, model.nv))
    qfrc_gravcomp = np.zeros((model.nv))
    for i in body_ids:
        body_weight = model.opt.gravity * model.body(i).mass
        mujoco.mj_jac(model, data, jac, None, data.body(i).xipos, i)
        q_weight = jac.T @ body_weight
        qfrc_gravcomp -= q_weight
    return qfrc_gravcomp


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data = mujoco.MjData(model)

    # Bodies for which to apply gravity compensation.
    left_bodies = mink.get_subtree_body_ids(model, model.body("left/base_link").id)
    right_bodies = mink.get_subtree_body_ids(model, model.body("right/base_link").id)
    gravcomp_bodies = np.concatenate([left_bodies, right_bodies])

    # Get the dof and actuator ids for the joints we wish to control.
    joint_names: list[str] = []
    velocity_limits: dict[str, float] = {}
    for prefix in ["left", "right"]:
        for n in _JOINT_NAMES:
            name = f"{prefix}/{n}"
            joint_names.append(name)
            velocity_limits[name] = _VELOCITY_LIMITS[n]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    configuration = mink.Configuration(model)

    tasks = [
        l_ee_task := mink.FrameTask(
            frame_name="left/gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        r_ee_task := mink.FrameTask(
            frame_name="right/gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model, cost=1e-4),
    ]

    # Enable collision avoidance between the following geoms:
    # geoms starting at subtree "right wrist" - "table",
    # geoms starting at subtree "left wrist"  - "table",
    # geoms starting at subtree "right wrist" - geoms starting at subtree "left wrist".
    l_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("left/wrist_link").id)
    r_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("right/wrist_link").id)
    l_geoms = mink.get_subtree_geom_ids(model, model.body("left/upper_arm_link").id)
    r_geoms = mink.get_subtree_geom_ids(model, model.body("right/upper_arm_link").id)
    frame_geoms = mink.get_body_geom_ids(model, model.body("metal_frame").id)
    collision_pairs = [
        (l_wrist_geoms, r_wrist_geoms),
        (l_geoms + r_geoms, frame_geoms + ["table"]),
    ]
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,  # type: ignore
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.1,
    )

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, velocity_limits),
        collision_avoidance_limit,
    ]

    l_mid = model.body("left/target").mocapid[0]
    r_mid = model.body("right/target").mocapid[0]
    solver = "quadprog"
    pos_threshold = 1e-2
    ori_threshold = 1e-2
    max_iters = 2

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        posture_task.set_target_from_configuration(configuration)

        # Initialize mocap targets at the end-effector site.
        mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
        mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")

        rate = RateLimiter(frequency=200.0)
        while viewer.is_running():
            # Update task targets.
            l_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
            r_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

            # Compute velocity and integrate into the next configuration.
            for i in range(max_iters):
                vel = mink.solve_ik(
                    configuration,
                    tasks,
                    rate.dt,
                    solver,
                    limits=limits,
                    damping=1e-5,
                )
                configuration.integrate_inplace(vel, rate.dt)

                l_err = l_ee_task.compute_error(configuration)
                l_pos_achieved = np.linalg.norm(l_err[:3]) <= pos_threshold
                l_ori_achieved = np.linalg.norm(l_err[3:]) <= ori_threshold
                r_err = l_ee_task.compute_error(configuration)
                r_pos_achieved = np.linalg.norm(r_err[:3]) <= pos_threshold
                r_ori_achieved = np.linalg.norm(r_err[3:]) <= ori_threshold
                if (
                    l_pos_achieved
                    and l_ori_achieved
                    and r_pos_achieved
                    and r_ori_achieved
                ):
                    break

            data.ctrl[actuator_ids] = configuration.q[dof_ids]
            data.qfrc_applied = gravcomp(model, data, gravcomp_bodies)
            mujoco.mj_step(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
