"""Googly eyes: two eyeballs in ball-and-socket joints track a moving target with
:class:`mink.LookAtTask`.

Each socket is a ball joint whose ``range`` bounds the eye's total rotation angle.
:class:`mink.ConfigurationLimit` enforces that bound, so when the target swings past
the socket range the eyes pin at the edge and reacquire on the way back. Gaze rays
turn red while an eye is at its limit.

Watch the eyes track the auto-orbiting target::

    uv run mjpython examples/googly_eyes.py

Drive the target yourself by dragging it in the viewer::

    uv run mjpython examples/googly_eyes.py --interactive

See what the limit buys you by disabling it (the eyes roll past their sockets)::

    uv run mjpython examples/googly_eyes.py --no-limit
"""

import argparse

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_SOCKET_RANGE = 0.7  # Max rotation angle of each eye, in radians.

_XML = f"""
<mujoco model="googly eyes">
  <compiler angle="radian"/>
  <default>
    <geom contype="0" conaffinity="0"/>
  </default>
  <statistic center="0.4 0 1.0" extent="2"/>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.25 0.25 0.25" specular="0 0 0"/>
    <global azimuth="160" elevation="-15"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2"
      width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="grid"/>
    <body name="face" pos="0 0 1.1">
      <geom name="post" type="cylinder" fromto="0 0 -1.1 0 0 -0.4" size="0.04"
        rgba="0.55 0.57 0.6 1"/>
      <geom name="plate" type="box" size="0.05 0.42 0.4" pos="0 0 -0.02"
        rgba="0.98 0.85 0.4 1"/>
      <geom name="left_brow" type="capsule" fromto="0.06 0.1 0.27 0.06 0.3 0.29"
        size="0.018" rgba="0.25 0.18 0.12 1"/>
      <geom name="right_brow" type="capsule" fromto="0.06 -0.1 0.27 0.06 -0.3 0.29"
        size="0.018" rgba="0.25 0.18 0.12 1"/>
      <geom name="mouth" type="capsule" fromto="0.06 0.13 -0.26 0.06 -0.13 -0.26"
        size="0.025" rgba="0.75 0.3 0.3 1"/>
      <body name="left_eye" pos="0.05 0.18 0.08">
        <joint name="left_eye" type="ball" range="0 {_SOCKET_RANGE}"/>
        <geom name="left_sclera" type="sphere" size="0.12" rgba="1 1 1 1"/>
        <geom name="left_iris" type="sphere" size="0.05" pos="0.08 0 0"
          rgba="0.35 0.55 0.85 1"/>
        <geom name="left_pupil" type="sphere" size="0.028" pos="0.105 0 0"
          rgba="0.05 0.05 0.05 1"/>
        <site name="left_eye" group="4"/>
      </body>
      <body name="right_eye" pos="0.05 -0.18 0.08">
        <joint name="right_eye" type="ball" range="0 {_SOCKET_RANGE}"/>
        <geom name="right_sclera" type="sphere" size="0.12" rgba="1 1 1 1"/>
        <geom name="right_iris" type="sphere" size="0.05" pos="0.08 0 0"
          rgba="0.35 0.55 0.85 1"/>
        <geom name="right_pupil" type="sphere" size="0.028" pos="0.105 0 0"
          rgba="0.05 0.05 0.05 1"/>
        <site name="right_eye" group="4"/>
      </body>
    </body>
    <body name="target" mocap="true" pos="1.0 0 1.1">
      <geom name="target" type="sphere" size="0.05" rgba="1 0.45 0.1 1"/>
    </body>
  </worldbody>
  <keyframe>
    <key name="home" qpos="1 0 0 0 1 0 0 0"/>
  </keyframe>
</mujoco>
"""

_EYES = ["left_eye", "right_eye"]


def orbit_target(t: float) -> np.ndarray:
    """A looping path that swings far around the face, well past the socket range."""
    azimuth = 2.0 * np.sin(0.45 * t)
    return np.array(
        [
            1.2 * np.cos(azimuth),
            1.2 * np.sin(azimuth),
            1.1 + 0.45 * np.sin(1.3 * t),
        ]
    )


def make_solver(model: mujoco.MjModel, limit_enabled: bool = True):
    """Build the configuration, tasks and a stepping closure."""
    configuration = mink.Configuration(model)
    configuration.update_from_keyframe("home")

    look_at_tasks = [
        mink.LookAtTask(
            frame_name=eye,
            frame_type="site",
            axis=(1.0, 0.0, 0.0),
            cost=1.0,
            # Levenberg-Marquardt damping suppresses solver chatter when the target
            # is outside the socket range and the residual is large but unreachable.
            lm_damping=10.0,
        )
        for eye in _EYES
    ]
    posture_task = mink.PostureTask(model, cost=1e-3)
    posture_task.set_target_from_configuration(configuration)
    tasks = [*look_at_tasks, posture_task]

    limits: list[mink.Limit] = [
        mink.VelocityLimit(model, dict.fromkeys(_EYES, np.full(3, 4.0)))
    ]
    if limit_enabled:
        limits.append(mink.ConfigurationLimit(model))

    def step(target_pos: np.ndarray, dt: float) -> None:
        for task in look_at_tasks:
            task.set_target(target_pos)
        vel = mink.solve_ik(
            configuration, tasks, dt, "daqp", damping=1e-3, limits=limits
        )
        configuration.integrate_inplace(vel, dt)

    return configuration, step


def eye_angles(model: mujoco.MjModel, q: np.ndarray) -> list[float]:
    """Rotation angle of each eye away from its rest orientation."""
    angles = []
    for eye in _EYES:
        padr = model.jnt_qposadr[model.joint(eye).id]
        quat = q[padr : padr + 4]
        angles.append(2.0 * np.arctan2(np.linalg.norm(quat[1:]), abs(quat[0])))
    return angles


def draw_gaze_rays(
    scene: mujoco.MjvScene,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q: np.ndarray,
    target: np.ndarray,
) -> None:
    """Draw a ray from each pupil, red when the eye is pinned at its socket limit."""
    for eye, angle in zip(_EYES, eye_angles(model, q)):
        if scene.ngeom >= scene.maxgeom:
            return
        pinned = angle > 0.98 * _SOCKET_RANGE
        rgba = (1.0, 0.1, 0.1, 1.0) if pinned else (1.0, 0.7, 0.2, 1.0)
        start = data.site_xpos[model.site(eye).id]
        gaze = data.site_xmat[model.site(eye).id].reshape(3, 3)[:, 0]
        end = start + gaze * np.linalg.norm(target - start)
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            np.array(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006, start, end)
        scene.ngeom += 1


def run_viewer(
    model: mujoco.MjModel, interactive: bool = False, limit_enabled: bool = True
) -> None:
    configuration, step = make_solver(model, limit_enabled)
    data = configuration.data
    target_mid = model.body("target").mocapid[0]

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        rate = RateLimiter(frequency=200.0, warn=False)
        elapsed = 0.0
        while viewer.is_running():
            if not interactive:
                data.mocap_pos[target_mid] = orbit_target(elapsed)
                elapsed += rate.dt
            target = data.mocap_pos[target_mid].copy()
            step(target, rate.dt)
            mujoco.mj_camlight(model, data)
            viewer.user_scn.ngeom = 0
            draw_gaze_rays(viewer.user_scn, model, data, configuration.q, target)
            viewer.sync()
            rate.sleep()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Drag the target yourself instead of watching it auto-orbit.",
    )
    parser.add_argument(
        "--no-limit",
        action="store_true",
        help="Disable the ConfigurationLimit: the eyes roll past their sockets.",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_string(_XML)
    run_viewer(model, interactive=args.interactive, limit_enabled=not args.no_limit)


if __name__ == "__main__":
    main()
