from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from dm_control.viewer import user_input
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "stanford_tidybot" / "scene.xml"


@dataclass
class KeyCallback:
    fix_base: bool = False
    pause: bool = False

    def __call__(self, key: int) -> None:
        if key == user_input.KEY_ENTER:
            self.fix_base = not self.fix_base
        elif key == user_input.KEY_SPACE:
            self.pause = not self.pause


def inverse_dynamics_controller(model, data, qpos_desired, Kp, Kd, dof_mask, ctrl_mask):
    """PD controller + inverse dynamics. Also known as computed torque control."""
    qacc_desired = (
        Kp * (qpos_desired[dof_mask] - data.qpos[dof_mask]) - Kd * data.qvel[dof_mask]
    )
    qacc_prev = data.qacc.copy()
    data.qacc[dof_mask] = qacc_desired
    mujoco.mj_inverse(model, data)
    tau = data.qfrc_inverse.copy()
    data.ctrl[ctrl_mask] = tau[ctrl_mask]
    data.qacc = qacc_prev  # Restore qacc.


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Joints we wish to control.
    # fmt: off
    joint_names = [
        # Base joints.
        "joint_x", "joint_y", "joint_th",
        # Arm joints.
        "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7",
    ]
    # fmt: on
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    configuration = mink.Configuration(model)

    # Posture task.
    posture_cost = np.zeros((model.nv,))
    posture_cost[3:] = 1e-3
    posture_task = mink.PostureTask(model, cost=posture_cost)

    # Damping task.
    damping_cost = np.zeros((model.nv,))
    damping_cost[:3] = 100
    damping_task = mink.DampingTask(model, damping_cost)

    # Frame tasks.
    end_effector_task = mink.FrameTask(
        frame_name="pinch_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    mobile_base_task = mink.FrameTask(
        frame_name="base_link",
        frame_type="body",
        position_cost=[1, 1, 0.0],
        orientation_cost=1,
    )

    tasks = [
        end_effector_task,
        mobile_base_task,
        posture_task,
    ]

    limits = [
        mink.ConfigurationLimit(model),
    ]

    # IK settings.
    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20
    base_mid = model.body("base_target").mocapid[0]
    key_callback = KeyCallback()

    Kp = np.asarray(
        [50_000.0] * 3 + [50_000.0] * 7,
    )
    Kd = np.asarray(
        [200.0] * 3 + [200.0] * 7,
    )

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        mujoco.mj_resetDataKeyframe(model, data, model.key("test").id)
        configuration.update(data.qpos)
        posture_task.set_target(model.key("home").qpos)
        mujoco.mj_forward(model, data)

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "pinch_site_target", "pinch_site", "site")
        mink.move_mocap_to_frame(model, data, "base_target", "base_link", "body")

        rate = RateLimiter(frequency=500.0, warn=False)
        dt = rate.period
        t = 0.0
        while viewer.is_running():
            # Update task targets.
            end_effector_task.set_target(
                mink.SE3.from_mocap_name(model, data, "pinch_site_target")
            )

            # Peturb the base target using a sinusoidal function in 1D.
            freq = 0.5
            noise = 0.06 * np.sin(2 * np.pi * freq * t)
            data.mocap_pos[base_mid, 0] = noise
            mobile_base_task.set_target(
                mink.SE3.from_mocap_name(model, data, "base_target")
            )

            # Compute velocity and integrate into the next configuration.
            for i in range(max_iters):
                vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-8)
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
                # data.ctrl[actuator_ids] = configuration.q[dof_ids]
                inverse_dynamics_controller(
                    model,
                    data,
                    configuration.q,
                    Kp,
                    Kd,
                    dof_ids,
                    actuator_ids,
                )
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
            t += dt
