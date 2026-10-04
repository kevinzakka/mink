from pathlib import Path

import mujoco
import mujoco.viewer
from _model_assets import load_example_spec
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_ARM_XML = _HERE / "ufactory_xarm7" / "scene.xml"
_HAND_XML = _HERE / "leap_hand" / "right_hand.xml"

fingers = ["tip_1", "tip_2", "tip_3", "th_tip"]

# fmt: off
HOME_QPOS = [
    # xarm.
    0, -.247, 0, .909, 0, 1.15644, 0,
    # leap.
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
]
# fmt: on


def construct_model() -> mujoco.MjModel:
    arm = load_example_spec("ufactory_xarm7", _ARM_XML)
    hand = load_example_spec("leap_hand", _HAND_XML)

    palm = hand.body("palm_lower")
    palm.quat[:] = (0, 1, 0, 0)
    palm.pos[:] = (0.065, -0.04, 0)
    site = arm.site("attachment_site")
    arm.attach(hand, prefix="leap_right/", site=site)

    home_key = arm.key("home")
    arm.delete(home_key)
    arm.add_key(name="home", qpos=HOME_QPOS)

    for finger in fingers:
        body = arm.worldbody.add_body(name=f"{finger}_target", mocap=True)
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.02,) * 3,
            contype=0,
            conaffinity=0,
            rgba=(0.6, 0.3, 0.3, 0.5),
        )

    return arm.compile()


if __name__ == "__main__":
    model = construct_model()

    configuration = mink.Configuration(model)

    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    posture_task = mink.PostureTask(model=model, cost=5e-2)

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

    tasks = [end_effector_task, posture_task, *finger_tasks]

    limits = [
        mink.ConfigurationLimit(model=model),
    ]

    # IK settings.
    solver = "daqp"
    model = configuration.model
    data = configuration.data

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)

        # Define a rectangular region centered about the origin of the initial
        # end-effector pose. This can be used to clamp the end_effector_task target
        # to ensure that the end-effector stays within this region.
        # Roll and pitch limits are also defined to keep the hand pointed downward.
        T_eef_init = configuration.get_transform_frame_to_world(
            "attachment_site", "site"
        )
        translation_init = T_eef_init.translation()
        x_limits = (translation_init[0] - 0.1, translation_init[0] + 0.1)
        y_limits = (translation_init[1] - 0.1, translation_init[1] + 0.1)
        z_limits = (translation_init[2] - 0.1, translation_init[2] + 0.1)
        rpy_init = T_eef_init.rotation().as_rpy_radians()
        roll_limits = (rpy_init.roll - 0.2, rpy_init.roll + 0.2)
        pitch_limits = (rpy_init.pitch - 0.2, rpy_init.pitch + 0.2)

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
        for finger in fingers:
            mink.move_mocap_to_frame(
                model, data, f"{finger}_target", f"leap_right/{finger}", "site"
            )

        T_palm_prev = configuration.get_transform_frame_to_world(
            "leap_right/palm_lower", "body"
        )

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Clamp the target within the limits and then update the end-effector task.
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            T_wt_clamped = T_wt.clamp(
                x_translation=x_limits,
                y_translation=y_limits,
                z_translation=z_limits,
                roll_radians=roll_limits,
                pitch_radians=pitch_limits,
            )
            end_effector_task.set_target(T_wt_clamped)

            # Update finger tasks.
            for finger, task in zip(fingers, finger_tasks):
                T_pm = configuration.get_transform(
                    f"{finger}_target", "body", "leap_right/palm_lower", "body"
                )
                task.set_target(T_pm)

            T_palm = configuration.get_transform_frame_to_world(
                "leap_right/palm_lower", "body"
            )
            T = T_palm @ T_palm_prev.inverse()
            T_palm_prev = T_palm.copy()
            for finger in fingers:
                mocap_id = model.body(f"{finger}_target").mocapid[0]
                T_w_mocap = mink.SE3.from_mocap_id(data, mocap_id)
                T_w_mocap_new = T @ T_w_mocap
                data.mocap_pos[mocap_id] = T_w_mocap_new.translation()
                data.mocap_quat[mocap_id] = T_w_mocap_new.rotation().wxyz

            # Compute velocity and integrate into the next configuration.
            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, damping=1e-3, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
