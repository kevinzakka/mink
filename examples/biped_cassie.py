"""IK with two four-bar linkages on Agility Cassie."""

from pathlib import Path

import mujoco
import mujoco.viewer
from _model_assets import load_example_spec
from loop_rate_limiters import RateLimiter

import mink
from mink.lie import SE3

_HERE = Path(__file__).parent
_XML = _HERE / "agility_cassie" / "scene.xml"


def main():
    model = load_example_spec("agility_cassie", _XML).compile()
    configuration = mink.Configuration(model)

    tasks = [
        pelvis_orientation_task := mink.FrameTask(
            frame_name="cassie-pelvis",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=10.0,
        ),
        posture_task := mink.PostureTask(model, cost=1.0),
        com_task := mink.ComTask(cost=200.0),
    ]

    # Note: By not providing `equality_name`, all equality constraints in the model
    # will be regulated.
    equality_task = mink.EqualityConstraintTask(
        model=model,
        cost=500.0,
        gain=0.5,
        lm_damping=1e-3,
    )
    tasks.append(equality_task)

    feet = ["left-foot", "right-foot"]
    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="body",  # Cassie uses body for feet, not sites.
            position_cost=200.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        )
        feet_tasks.append(task)
    tasks.extend(feet_tasks)

    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]

    model = configuration.model
    data = configuration.data
    solver = "daqp"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        configuration.update_from_keyframe("home")
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)
        for i, foot in enumerate(feet):
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "body")
        data.mocap_pos[com_mid] = data.subtree_com[1]

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            com_task.set_target(data.mocap_pos[com_mid])
            for i, foot_task in enumerate(feet_tasks):
                foot_task.set_target(SE3.from_mocap_id(data, feet_mid[i]))

            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, damping=1e-1)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()
