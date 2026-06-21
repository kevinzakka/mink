"""UR5e with a wrist-mounted Intel RealSense D435i that keeps a moving target
centered in its field of view using :class:`mink.LookAtTask`.

The look-at task only constrains *where* the camera points, leaving roll about the
optical axis free, so the arm naturally orients the sensor toward the target while
the rest of the body is regularized by a posture task.

Watch the arm track the auto-orbiting target (press Tab to look through
``wrist_cam``)::

    mjpython examples/arm_ur5e_wrist_cam_lookat.py

Drive the target yourself by dragging it in the viewer::

    mjpython examples/arm_ur5e_wrist_cam_lookat.py --interactive

Record a side-by-side (third-person | onboard) video::

    python examples/arm_ur5e_wrist_cam_lookat.py --record lookat.mp4
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_UR5E = _HERE / "universal_robots_ur5e" / "scene_wrist_cam.xml"
_REALSENSE = _HERE / "realsense_d435i.xml"

# Pose of the camera/sensor on the wrist. The quaternion rotates +90 deg about the
# local x-axis so that the frame's -z axis (the MuJoCo camera optical axis) points
# outward along the wrist's +y, away from the arm. The sensor sits past the flange
# (at y=0.1) so it is visible rather than buried in the wrist link.
_MOUNT_QUAT = (1.0, 1.0, 0.0, 0.0)
_SENSOR_POS = (0.0, 0.12, 0.0)
_CAM_POS = (0.0, 0.145, 0.0)

# Where the camera looks by default, in the world frame. Placed in front of the
# robot; the initial pose is solved to already point at it (see make_solver).
_DEFAULT_TARGET = (0.85, 0.0, 0.6)

_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


def build_model() -> mujoco.MjModel:
    """Assemble the UR5e scene with a wrist-mounted RealSense camera."""
    spec = mujoco.MjSpec.from_file(_UR5E.as_posix())
    wrist = spec.body("wrist_3_link")

    realsense = mujoco.MjSpec.from_file(_REALSENSE.as_posix())
    frame = wrist.add_frame(pos=_SENSOR_POS, quat=_MOUNT_QUAT)
    frame.attach_body(realsense.body("d435i"), "rs_", "")

    # Co-located camera and site sharing the optical frame: the site is what the
    # look-at task controls, the camera is what we render the onboard view from.
    wrist.add_camera(name="wrist_cam", pos=_CAM_POS, quat=_MOUNT_QUAT, fovy=70)
    wrist.add_site(
        name="wrist_cam_site", pos=_CAM_POS, quat=_MOUNT_QUAT, size=[0.005] * 3, group=4
    )
    model = spec.compile()
    # Reset the target mocap to its default spot (Backspace restores body_pos).
    model.body_pos[model.body("target").id] = _DEFAULT_TARGET
    return model


def orbit_target(t: float) -> np.ndarray:
    """A looping path in front of the robot for the target to follow."""
    return np.array(
        [0.7 + 0.3 * np.cos(t), 0.6 * np.sin(t), 0.55 + 0.25 * np.sin(2.0 * t)]
    )


def _build_tasks(model: mujoco.MjModel, configuration: mink.Configuration):
    """Build the look-at and posture tasks plus the limits."""
    look_at_task = mink.LookAtTask(
        frame_name="wrist_cam_site",
        frame_type="site",
        axis=(0.0, 0.0, -1.0),  # MuJoCo camera optical axis.
        cost=1.0,
        lm_damping=1e-3,
    )
    # Let the whole arm reorient to follow the target (like turning your body to
    # keep something in view) rather than pinning the camera in place: a fixed
    # position would force the wrist through singularities. The posture task keeps
    # the arm near its rest pose, which also keeps a natural standoff from the
    # target so the camera never drives onto it (where look-at is ill-conditioned).
    posture_task = mink.PostureTask(model, cost=1e-2)
    posture_task.set_target_from_configuration(configuration)

    # Cap joint velocities so the camera slews smoothly to a new target instead of
    # snapping in a single step.
    max_velocities = dict.fromkeys(_JOINTS, np.pi)
    limits = [
        mink.ConfigurationLimit(model),
        mink.VelocityLimit(model, max_velocities),
    ]
    return [look_at_task, posture_task], limits


def make_solver(model: mujoco.MjModel):
    """Build the configuration, tasks and a stepping closure.

    Returns ``(configuration, step, initial_qpos)``. ``initial_qpos`` is a pose
    that already looks at the default target, solved once from the home keyframe;
    the caller restores it on reset (see :func:`run_interactive`).
    """
    configuration = mink.Configuration(model)
    configuration.update_from_keyframe("home")
    tasks, limits = _build_tasks(model, configuration)
    look_at_task, posture_task = tasks

    # Solve once so the arm starts already looking at the default target instead
    # of slewing to acquire it on the first frame.
    look_at_task.set_target(np.array(_DEFAULT_TARGET))
    for _ in range(400):
        vel = mink.solve_ik(
            configuration, tasks, 1 / 200, "daqp", damping=1e-3, limits=limits
        )
        configuration.integrate_inplace(vel, 1 / 200)
    initial_qpos = configuration.q
    # Anchor the posture target at this locked-on viewing pose.
    posture_task.set_target_from_configuration(configuration)

    def step(target_pos: np.ndarray, dt: float) -> None:
        look_at_task.set_target(target_pos)
        vel = mink.solve_ik(
            configuration, tasks, dt, "daqp", damping=1e-3, limits=limits
        )
        configuration.integrate_inplace(vel, dt)

    return configuration, step, initial_qpos


def _downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean downsample an RGB image by an integer factor."""
    h, w = (image.shape[0] // factor) * factor, (image.shape[1] // factor) * factor
    blocks = image[:h, :w].reshape(h // factor, factor, w // factor, factor, 3)
    return blocks.mean(axis=(1, 3)).astype(np.uint8)


def draw_gaze_ray(scene: mujoco.MjvScene, start: np.ndarray, end: np.ndarray) -> None:
    """Draw a thin line from the camera to the target into a viewer scene."""
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.array([1.0, 0.5, 0.0, 1.0], dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004, start, end)
    scene.ngeom += 1


def run_viewer(model: mujoco.MjModel, interactive: bool = False) -> None:
    configuration, step, initial_qpos = make_solver(model)
    data = configuration.data
    target_mid = model.body("target").mocapid[0]
    cam_site_id = model.site("wrist_cam_site").id

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        rate = RateLimiter(frequency=200.0, warn=False)
        elapsed = 0.0
        while viewer.is_running():
            # The viewer's reset (Backspace) calls mj_resetData, which sends every
            # joint to qpos0 (the model reference pose) -- here, the arm fully
            # extended and pointing away. Detect that and restore the locked-on
            # viewing pose instead of letting the IK slew out of the singularity.
            if np.allclose(data.qpos, model.qpos0):
                configuration.update(initial_qpos)
                elapsed = 0.0
            if not interactive:
                # Drive the target along a fixed orbit so the demo runs hands-free.
                data.mocap_pos[target_mid] = orbit_target(elapsed)
                elapsed += rate.dt
            step(data.mocap_pos[target_mid].copy(), rate.dt)
            mujoco.mj_camlight(model, data)
            viewer.user_scn.ngeom = 0
            draw_gaze_ray(
                viewer.user_scn, data.site_xpos[cam_site_id], data.mocap_pos[target_mid]
            )
            viewer.sync()
            rate.sleep()


def run_record(model: mujoco.MjModel, path: Path, duration: float, fps: int) -> None:
    configuration, step, _ = make_solver(model)
    data = configuration.data
    target_mid = model.body("target").mocapid[0]

    third = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, third)
    third.distance, third.azimuth, third.elevation = 2.2, 130.0, -20.0
    third.lookat[:] = [0.3, 0.0, 0.4]

    h, w = 480, 640
    scale, margin = 3, 16  # Onboard inset: 1/scale size, inset from the corner.
    dt = 1.0 / fps
    frames = []
    with mujoco.Renderer(model, h, w) as renderer:
        for i in range(int(duration * fps)):
            target = orbit_target(i * dt)
            data.mocap_pos[target_mid] = target
            step(target, dt)
            mujoco.mj_camlight(model, data)

            renderer.update_scene(data, camera=third)
            frame = renderer.render().copy()
            renderer.update_scene(data, camera="wrist_cam")
            inset = _downsample(renderer.render(), scale)

            # Nest the onboard view in the bottom-right corner with a white border.
            ih, iw = inset.shape[:2]
            y0, x0 = h - ih - margin, w - iw - margin
            frame[y0 - 2 : y0 + ih + 2, x0 - 2 : x0 + iw + 2] = 255
            frame[y0 : y0 + ih, x0 : x0 + iw] = inset
            frames.append(frame)

    path = Path(path)
    if path.suffix == ".gif":
        iio.imwrite(path, frames, loop=0, fps=fps)
    else:
        iio.imwrite(path, frames, fps=fps)
    print(f"Wrote {len(frames)} frames to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="Render a side-by-side video to this path (.mp4 or .gif) instead of "
        "launching the interactive viewer.",
    )
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Drag the target yourself instead of watching it auto-orbit.",
    )
    args = parser.parse_args()

    model = build_model()
    if args.record is not None:
        run_record(model, args.record, args.duration, args.fps)
    else:
        run_viewer(model, interactive=args.interactive)


if __name__ == "__main__":
    main()
