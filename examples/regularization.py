import argparse
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "universal_robots_ur5e" / "scene_plain.xml"
_MAX_TRACE_POINTS = 250


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arm IIWA example with configurable regularization weight."
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

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=0.0,
        ),
        mink.KineticEnergyRegularizationTask(cost=args.energy_reg),
    ]

    solver = "daqp"
    model = configuration.model
    data = configuration.data

    # Initialize trace storage
    times = deque(maxlen=_MAX_TRACE_POINTS)
    positions = deque(maxlen=_MAX_TRACE_POINTS)

    def add_visual_capsule(scene, point1, point2, radius, rgba):
        """Adds one capsule to an mjvScene."""
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
            rgba = np.array([0, 1, 0.5, 0.8])
            radius = 0.003
            add_visual_capsule(scn, positions[i], positions[i + 1], radius, rgba)

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        configuration.update_from_keyframe("home")

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        rate = RateLimiter(frequency=50.0, warn=False)
        t = 0.0
        while viewer.is_running():
            # Update task target using a figure-8 pattern.
            x = 0.5 + 0.1 * np.sin(2 * t)
            y = 0.4 * np.sin(t)
            data.mocap_pos[0] = np.array([x, y, 0.25])
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            # Compute velocity and integrate into the next configuration.
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            site_pos = data.site_xpos[data.site("attachment_site").id].copy()
            positions.append(site_pos)
            times.append(t)
            modify_scene(viewer.user_scn)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
            t += rate.dt
