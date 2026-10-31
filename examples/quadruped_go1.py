from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from _model_assets import load_example_spec
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "unitree_go1" / "scene.xml"


def cubic_bezier_interpolation(y_start, y_end, x):
    y_diff = y_end - y_start
    bezier = x**3 + 3 * (x**2 * (1 - x))
    return y_start + y_diff * bezier


def get_foot_z(phi: np.ndarray, swing_height: float = 0.08) -> np.ndarray:
    x = (phi + np.pi) / (2 * np.pi)  # [0, 1].
    x = np.clip(x, 0, 1)
    return np.where(
        x <= 0.5,
        cubic_bezier_interpolation(0, swing_height, 2 * x),  # Swing
        cubic_bezier_interpolation(swing_height, 0, 2 * x - 1),  # Stance
    )


if __name__ == "__main__":
    model = load_example_spec("unitree_go1", _XML).compile()

    configuration = mink.Configuration(model)

    feet = ["FL", "FR", "RR", "RL"]

    base_task = mink.FrameTask(
        frame_name="trunk",
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    posture_task = mink.PostureTask(model, cost=1e-5)

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
        )
        feet_tasks.append(task)

    tasks = [base_task, posture_task, *feet_tasks]

    base_mid = model.body("trunk_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]

    model = configuration.model
    data = configuration.data
    solver = "daqp"

    # Gait parameters.
    # Initial phase offsets for each foot
    foot_phases = np.array([np.pi, 0, np.pi, 0])
    swing_height = 0.1  # m.
    gait_freq = 1.5  # Hz.
    rate = RateLimiter(frequency=500.0, warn=False)
    phase_dt = 2 * np.pi * gait_freq * rate.dt

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("home")
        posture_task.set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective sites.
        for foot in feet:
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
        mink.move_mocap_to_frame(model, data, "trunk_target", "trunk", "body")

        # Store initial relative positions of feet to the base.
        initial_feet_offsets = []
        for i in range(4):
            offset = data.mocap_pos[feet_mid[i], :2] - data.mocap_pos[base_mid, :2]
            initial_feet_offsets.append(offset)

        time = 0.0
        while viewer.is_running():
            # Update task targets.
            # Make the base x-y track a circle.
            base_pos = np.array([0.5 * np.cos(time), 0.5 * np.sin(time)])
            data.mocap_pos[base_mid, :2] = base_pos

            base_task.set_target(mink.SE3.from_mocap_id(data, base_mid))

            # Get foot height based on current phase
            z = get_foot_z(foot_phases, swing_height)
            for i in range(4):
                # Use the stored offsets to maintain relative foot positions
                data.mocap_pos[feet_mid[i], :2] = base_pos + initial_feet_offsets[i]
                data.mocap_pos[feet_mid[i], 2] = z[i]
            for i, task in enumerate(feet_tasks):
                task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))

            # Compute velocity, integrate and set control signal.
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, damping=1e-5)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
            time += rate.dt

            # Increment phase.
            foot_phases = (foot_phases + phase_dt) % (2 * np.pi)
            foot_phases = foot_phases - np.where(foot_phases > np.pi, 2 * np.pi, 0)
