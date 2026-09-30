"""UR5e end-effector tracking with an elastic constraint, :class:`mink.Elastic`.

The end-effector task is passed as an elastic constraint: instead of a quadratic cost
that competes with the posture regularizer, it is an exact L1 penalty. While the
target is reachable, the end effector tracks it exactly, as if the task were a hard
constraint, even though the posture task keeps pulling toward the home
configuration. Drag the target out of
the workspace (or into a pose the joint limits cannot reach) and the task yields
instead of making the QP infeasible. The orientation penalty is lower than the
position penalty, so orientation gives way first and position is held as long as
possible.

The target turns green while the task is held exactly and red while it yields.

    uv run mjpython examples/arm_ur5e_elastic.py
"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "universal_robots_ur5e" / "scene.xml"

# Per-component L1 penalties: (x, y, z, roll, pitch, yaw).
_PENALTY = np.array([1e3, 1e3, 1e3, 1e1, 1e1, 1e1])

# Linearized task residual above which the task is considered to be yielding.
_YIELD_THRESHOLD = 1e-6

_RED_COLOR = (0.6, 0.3, 0.3, 0.2)
_GREEN_COLOR = (0.3, 0.6, 0.3, 0.2)


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)

    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    constraints = [mink.Elastic(end_effector_task, penalty=_PENALTY)]
    posture_task = mink.PostureTask(model, cost=1e-1)
    tasks = [posture_task]

    # Enable collision avoidance between the following geoms:
    collision_pairs = [
        (["wrist_3_link"], ["floor", "wall"]),
    ]

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs),
    ]

    max_velocities = {
        "shoulder_pan": np.pi,
        "shoulder_lift": np.pi,
        "elbow": np.pi,
        "wrist_1": np.pi,
        "wrist_2": np.pi,
        "wrist_3": np.pi,
    }
    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)

    target_geom_id = model.body("target").geomadr[0]
    model = configuration.model
    data = configuration.data
    solver = "daqp"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("home")
        posture_task.set_target(configuration.q)

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update task target.
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            # Compute velocity. The elastic constraint is held exactly when its
            # linearized residual J dq + e is zero.
            vel = mink.solve_ik(
                configuration,
                tasks,
                rate.dt,
                solver,
                limits=limits,
                constraints=constraints,
            )
            residual = end_effector_task.compute_jacobian(
                configuration
            ) @ vel * rate.dt + end_effector_task.compute_error(configuration)
            yielding = bool(np.linalg.norm(residual) > _YIELD_THRESHOLD)

            # Integrate into the next configuration.
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Green = held exactly, red = yielding.
            with viewer.lock():
                model.geom_rgba[target_geom_id] = (
                    _RED_COLOR if yielding else _GREEN_COLOR
                )

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
