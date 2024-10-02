from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from dm_control import mjcf
from dm_control.viewer import user_input
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_ARM_XML = _HERE / "stanford_tidybot" / "scene_mobile_kinova.xml"
_HAND_XML = _HERE / "leap_hand" / "right_hand.xml"

fingers = ["tip_1", "tip_2", "tip_3", "th_tip"]

# fmt: off
HOME_QPOS = [
    # Mobile Base.
    0, 0, 0,
    # Kinova.
    0, 0.26179939, 3.14159265, -2.26892803, 0, 0.95993109, 1.57079633,
    # Leap hand.
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
]
# fmt: on


def construct_model():
    arm_mjcf = mjcf.from_path(_ARM_XML.as_posix())
    arm_mjcf.find("key", "home").remove()
    arm_mjcf.find("key", "retract").remove()

    hand_mjcf = mjcf.from_path(_HAND_XML.as_posix())
    palm = hand_mjcf.worldbody.find("body", "palm_lower")
    palm.quat = (0, 0.707, 0.707, 0)
    palm.pos = (0.03, 0.06, -0.0925)
    attach_site = arm_mjcf.worldbody.find("site", "pinch_site")
    attach_site.attach(hand_mjcf)

    arm_mjcf.keyframe.add("key", name="home", qpos=HOME_QPOS)

    for finger in fingers:
        body = arm_mjcf.worldbody.add("body", name=f"{finger}_target", mocap=True)
        body.add(
            "geom",
            type="sphere",
            size=".02",
            contype="0",
            conaffinity="0",
            rgba=".6 .3 .3 .5",
        )

    return mujoco.MjModel.from_xml_string(
        arm_mjcf.to_xml_string(), arm_mjcf.get_assets()
    )


@dataclass
class KeyCallback:
    fix_base: bool = False
    pause: bool = False

    def __call__(self, key: int) -> None:
        if key == user_input.KEY_ENTER:
            self.fix_base = not self.fix_base
        elif key == user_input.KEY_SPACE:
            self.pause = not self.pause


if __name__ == "__main__":
    model = construct_model()

    configuration = mink.Configuration(model)

    end_effector_task = mink.FrameTask(
        frame_name="pinch_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    # When move the base, mainly focus on the motion on xy plane, minimize the rotation.
    posture_cost = np.zeros((model.nv,))
    posture_cost[2] = 1e-3  # Mobile Base.
    # posture_cost[-16:] = 5e-2  # Leap Hand.
    posture_cost[-16:] = 1e-3  # Leap Hand.

    posture_task = mink.PostureTask(model, cost=posture_cost)

    immobile_base_cost = np.zeros((model.nv,))
    immobile_base_cost[:2] = 100
    immobile_base_cost[2] = 1e-3
    damping_task = mink.DampingTask(model, immobile_base_cost)

    finger_tasks = []
    for finger in fingers:
        task = mink.RelativeFrameTask(
            frame_name=f"leap_right/{finger}",
            frame_type="site",
            root_name="leap_right/palm_lower",
            root_type="body",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1e-3,
        )
        finger_tasks.append(task)

    tasks = [
        end_effector_task,
        posture_task,
        *finger_tasks,
    ]

    limits = [
        mink.ConfigurationLimit(model),
    ]

    # IK settings.
    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20
    model = configuration.model
    data = configuration.data

    key_callback = KeyCallback()

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "pinch_site_target", "pinch_site", "site")
        for finger in fingers:
            mink.move_mocap_to_frame(
                model, data, f"{finger}_target", f"leap_right/{finger}", "site"
            )

        T_eef_prev = configuration.get_transform_frame_to_world("pinch_site", "site")

        rate = RateLimiter(frequency=50.0, warn=False)
        dt = rate.period
        t = 0.0
        while viewer.is_running():
            # Update task target.
            T_wt = mink.SE3.from_mocap_name(model, data, "pinch_site_target")
            end_effector_task.set_target(T_wt)

            # Update finger tasks.
            for finger, task in zip(fingers, finger_tasks):
                T_pm = configuration.get_transform(
                    f"{finger}_target", "body", "leap_right/palm_lower", "body"
                )
                task.set_target(T_pm)

            for finger in fingers:
                T_eef = configuration.get_transform_frame_to_world("pinch_site", "site")
                T = T_eef @ T_eef_prev.inverse()
                T_w_mocap = mink.SE3.from_mocap_name(model, data, f"{finger}_target")
                T_w_mocap_new = T @ T_w_mocap
                data.mocap_pos[model.body(f"{finger}_target").mocapid[0]] = (
                    T_w_mocap_new.translation()
                )
                data.mocap_quat[model.body(f"{finger}_target").mocapid[0]] = (
                    T_w_mocap_new.rotation().wxyz
                )

            # Compute velocity and integrate into the next configuration.
            for i in range(max_iters):
                if key_callback.fix_base:
                    vel = mink.solve_ik(
                        configuration, [*tasks, damping_task], rate.dt, solver, 1e-3
                    )
                else:
                    vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3)
                configuration.integrate_inplace(vel, rate.dt)

                # Exit condition.
                pos_achieved = True
                ori_achieved = True
                err = end_effector_task.compute_error(configuration)
                pos_achieved &= bool(np.linalg.norm(err[:3]) <= pos_threshold)
                ori_achieved &= bool(np.linalg.norm(err[3:]) <= ori_threshold)
                if pos_achieved and ori_achieved:
                    break

            if not key_callback.pause:
                data.ctrl = configuration.q
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)

            T_eef_prev = T_eef.copy()

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
            t += dt
