from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink
from mink.lie import SE3, SO3
from mink.utils import set_mocap_pose_from_site

_HERE = Path(__file__).parent
_XML = _HERE / "kuka_iiwa_14" / "scene.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    ## =================== ##
    ## Setup IK.
    ## =================== ##

    configuration = mink.Configuration(model)

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1e-2,
        ),
        posture_task := mink.PostureTask(model=model, cost=1e-1),
    ]

    limits = [
        mink.ConfigurationLimit(model=configuration.model),
    ]

    ## =================== ##

    mid = model.body("target").mocapid[0]

    # IK settings.
    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        # Initialize the mocap target at the end-effector site.
        set_mocap_pose_from_site(model, data, "target", "attachment_site")

        posture_task.set_target_from_configuration(configuration)

        rate = RateLimiter(frequency=500.0)
        while viewer.is_running():
            target_pos = data.mocap_pos[mid]
            target_ori = data.mocap_quat[mid]
            target_pose = SE3.from_rotation_and_translation(SO3(target_ori), target_pos)
            end_effector_task.set_target(target_pose)

            # Compute velocity and integrate into the next configuration.
            for i in range(max_iters):
                vel = mink.solve_ik(configuration, tasks, limits, rate.dt, solver, 1e-3)
                configuration.integrate_inplace(vel, rate.dt)
                err = end_effector_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
                if pos_achieved and ori_achieved:
                    print(f"Exiting after {i} iterations.")
                    break

            data.ctrl = configuration.q
            mujoco.mj_step(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
