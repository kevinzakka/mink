"""Apollo humanoid visually tracking a target with its gaze.

The robot keeps its feet planted and balances (center-of-mass task) while the
look-at task points its head at a moving target. Because the gaze frame sits at
the end of the 3-DOF neck and the torso is left free, the look-at recruits the
whole upper body: the head pans and tilts and the waist turns to follow the
target, the way a person turns to watch something move.

Watch the robot track the auto-orbiting target::

    mjpython examples/humanoid_apollo.py

Drive the target yourself by dragging it in the viewer::

    mjpython examples/humanoid_apollo.py --interactive

Record a video::

    python examples/humanoid_apollo.py --record gaze.mp4
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
_XML = _HERE / "apptronik_apollo" / "scene_gaze.xml"

_FEET = ["left_foot", "right_foot"]
_HANDS = ["left_palm", "right_palm"]


def build_model() -> mujoco.MjModel:
    """Load the Apollo gaze scene and add a gaze frame at the end of the neck."""
    spec = mujoco.MjSpec.from_file(_XML.as_posix())
    spec.body("neck_pitch_link").add_site(
        name="gaze", pos=[0.08, 0.0, 0.0], size=[0.02] * 3, group=4
    )
    return spec.compile()


def orbit_target(t: float, face_pos: np.ndarray) -> np.ndarray:
    """A figure-eight in front of the face for the gaze to follow."""
    w = 2.0 * np.pi / 8.0
    return face_pos + np.array([1.0, 1.1 * np.sin(w * t), 0.45 * np.sin(2.0 * w * t)])


def _frame_camera(cam: mujoco.MjvCamera) -> None:
    """A 3/4 view that keeps both the robot and the orbiting target in frame."""
    cam.lookat[:] = [0.5, 0.0, 1.2]
    cam.distance = 3.6
    cam.azimuth = 135.0
    cam.elevation = -15.0


def make_solver(model: mujoco.MjModel):
    """Build the configuration, tasks and a stepping closure.

    Returns ``(configuration, step, initial_qpos, face_pos)``. ``face_pos`` is the
    world position of the gaze frame at the initial pose, used to center the orbit.
    """
    configuration = mink.Configuration(model)
    configuration.update_from_keyframe("stand")

    pelvis_task = mink.FrameTask(
        "base_link", "body", position_cost=0.0, orientation_cost=1.0, lm_damping=1.0
    )
    posture_task = mink.PostureTask(model, cost=1e-1)
    com_task = mink.ComTask(cost=10.0)
    # Feet planted, hands resting -- they hold their initial pose while the gaze
    # task moves the upper body. The torso is deliberately *not* constrained, so
    # the waist is free to turn and help the neck reach wide targets.
    feet_tasks = [
        mink.FrameTask(
            f, "site", position_cost=100.0, orientation_cost=1.0, lm_damping=1.0
        )
        for f in _FEET
    ]
    hand_tasks = [
        mink.FrameTask(
            h, "site", position_cost=5.0, orientation_cost=1.0, lm_damping=1.0
        )
        for h in _HANDS
    ]

    # The gaze frame's +x points up-and-forward at the neutral pose. Derive the
    # local axis that is *level* forward (world +x, the robot's facing direction)
    # so aligning it with the target makes the face point straight at it rather
    # than tilting too low.
    gaze_rot = configuration.get_transform_frame_to_world("gaze", "site").rotation()
    gaze_axis = gaze_rot.as_matrix().T @ np.array([1.0, 0.0, 0.0])
    gaze_task = mink.LookAtTask(
        "gaze", "site", axis=gaze_axis, cost=1.0, lm_damping=1.0
    )

    tasks = [pelvis_task, posture_task, com_task, *feet_tasks, *hand_tasks, gaze_task]
    for task in (pelvis_task, posture_task, *feet_tasks, *hand_tasks):
        task.set_target_from_configuration(configuration)
    com_task.set_target(configuration.data.subtree_com[1].copy())

    limits = [mink.ConfigurationLimit(model)]

    initial_qpos = configuration.q
    face_pos = configuration.get_transform_frame_to_world("gaze", "site").translation()

    def step(target_pos: np.ndarray, dt: float) -> None:
        gaze_task.set_target(target_pos)
        vel = mink.solve_ik(
            configuration, tasks, dt, "daqp", damping=1e-1, limits=limits
        )
        configuration.integrate_inplace(vel, dt)

    return configuration, step, initial_qpos, face_pos


def run_viewer(model: mujoco.MjModel, interactive: bool = False) -> None:
    configuration, step, initial_qpos, face_pos = make_solver(model)
    data = configuration.data
    target_mid = model.body("head_target").mocapid[0]
    data.mocap_pos[target_mid] = orbit_target(0.0, face_pos)

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        _frame_camera(viewer.cam)
        rate = RateLimiter(frequency=200.0, warn=False)
        elapsed = 0.0
        while viewer.is_running():
            # The viewer's reset (Backspace) sends every joint to qpos0; restore the
            # balanced standing pose instead of solving out of that singular state.
            if np.allclose(data.qpos, model.qpos0):
                configuration.update(initial_qpos)
                elapsed = 0.0
            if not interactive:
                data.mocap_pos[target_mid] = orbit_target(elapsed, face_pos)
                elapsed += rate.dt
            step(data.mocap_pos[target_mid].copy(), rate.dt)
            mujoco.mj_camlight(model, data)
            viewer.sync()
            rate.sleep()


def run_record(model: mujoco.MjModel, path: Path, duration: float, fps: int) -> None:
    configuration, step, _, face_pos = make_solver(model)
    data = configuration.data
    target_mid = model.body("head_target").mocapid[0]

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    _frame_camera(cam)

    dt = 1.0 / fps
    frames = []
    with mujoco.Renderer(model, 720, 960) as renderer:
        for i in range(int(duration * fps)):
            target = orbit_target(i * dt, face_pos)
            data.mocap_pos[target_mid] = target
            step(target, dt)
            mujoco.mj_camlight(model, data)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

    path = Path(path)
    if path.suffix == ".gif":
        iio.imwrite(path, frames, loop=0, fps=fps)
    else:
        iio.imwrite(path, frames, fps=fps)
    print(f"Wrote {len(frames)} frames to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=16.0)
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
