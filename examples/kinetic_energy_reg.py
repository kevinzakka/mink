"""UR5e figure-8 path with kinetic energy regularization.

Example usage:

    python examples/kinetic_energy_reg.py --help
    python examples/kinetic_energy_reg.py --energy_reg 0.0   # No regularization.
    python examples/kinetic_energy_reg.py --energy_reg 1e-5  # Low regularization.
"""

import argparse
from collections import deque
from pathlib import Path
from typing import Deque

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "universal_robots_ur5e" / "scene_plain.xml"

# IK integration timestep, in [s].
_DT = 0.02

# Maximum number of trace points to plot.
_DESIRED_TRACE_DURATION = 5.0  # [s]
_MAX_TRACE_POINTS = int(_DESIRED_TRACE_DURATION / _DT)

# Trace visualization parameters.
_RGBA = np.array([0, 1, 0.5, 0.8])
_RADIUS = 0.003


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UR5e figure-8 path with kinetic energy regularization."
    )
    parser.add_argument(
        "--energy_reg",
        type=float,
        default=0.0,
        help="Regularization weight for the kinetic energy task.",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    configuration = mink.Configuration(model)

    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.0,
    )

    kinetic_energy_task = mink.KineticEnergyRegularizationTask(cost=args.energy_reg)
    kinetic_energy_task.set_dt(_DT)  # NOTE: This is required!

    # For storing and visualizing the end-effector path.
    positions: Deque[np.ndarray] = deque(maxlen=_MAX_TRACE_POINTS)

    def add_visual_capsule(scene, point1, point2, radius, rgba):
        if scene.ngeom >= scene.maxgeom:
            return
        scene.ngeom += 1
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom - 1],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            rgba.astype(np.float32),
        )
        mujoco.mjv_connector(
            scene.geoms[scene.ngeom - 1],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            point1,
            point2,
        )

    def modify_scene(scn):
        scn.ngeom = 0
        for i in range(len(positions) - 1):
            if np.allclose(positions[i], positions[i + 1]):
                continue
            add_visual_capsule(scn, positions[i], positions[i + 1], _RADIUS, _RGBA)

    solver = "daqp"
    model = configuration.model
    data = configuration.data

    # Do an initial solve to find the initial configuration that achieves the target
    # position.
    z = 0.25
    pos0 = np.array([0.5, 0.0, z])
    configuration.update_from_keyframe("home")
    end_effector_task.set_target(mink.SE3.from_translation(pos0))
    for _ in range(10):
        vel = mink.solve_ik(configuration, [end_effector_task], _DT, solver)
        configuration.integrate_inplace(vel, _DT)
    qpos0 = configuration.q.copy()

    tasks = [end_effector_task, kinetic_energy_task]

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        configuration.update(qpos0)
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        rate = RateLimiter(frequency=(1.0 / _DT), warn=False)
        t = 0.0
        while viewer.is_running():
            # Update task target using a figure-8 pattern.
            x = 0.5 + 0.1 * np.sin(2 * t)
            y = 0.2 * np.sin(t)
            data.mocap_pos[0] = np.array([x, y, z])
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            # Compute velocity and integrate into the next configuration.
            vel = mink.solve_ik(configuration, tasks, _DT, solver)
            configuration.integrate_inplace(vel, _DT)
            mujoco.mj_camlight(model, data)
            positions.append(data.site_xpos[data.site("attachment_site").id].copy())

            # Visualize at fixed FPS.
            modify_scene(viewer.user_scn)
            viewer.sync()
            rate.sleep()
            t += _DT
