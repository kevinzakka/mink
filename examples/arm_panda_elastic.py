"""Three Pandas track one target: as a task, a hard constraint, and an elastic
constraint.

Each arm also has a posture task pulling it toward home. The target sweeps forward
out of reach and back:

- Task (left, teal): the end-effector task is in ``tasks``. It never fails, but the
  posture task holds it short of the target.
- Hard constraint (middle): the task is in ``constraints``. It tracks exactly (blue)
  until the target pose becomes unreachable. The QP is then infeasible, and the arm
  stops (red) until the target passes back through its pose.
- Elastic (right): the task is wrapped in :class:`mink.Elastic`. It tracks exactly
  (blue) while it can. When it cannot (orange), orientation gives way first so that
  position stays exact as long as possible, and tracking resumes once the target
  returns.

    uv run mjpython examples/arm_panda_elastic.py
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_SCENE = _HERE / "franka_emika_panda" / "mjx_scene.xml"
_ARM = _HERE / "franka_emika_panda" / "mjx_panda.xml"

_MODES = ("task", "hard", "elastic")
_SPACING = 0.8  # Lateral distance between arm bases [m].
_POSTURE_COST = 0.3  # Exaggerated so the task arm's offset is easy to see.
_MAX_VELOCITY = np.pi  # [rad/s]

# Penalties for (x, y, z, roll, pitch, yaw). Orientation gets the lower one so it
# gives way first and position is held as long as possible.
_PENALTY = [1e3, 1e3, 1e3, 1e1, 1e1, 1e1]

# The target sweeps this far forward from the home pose and back [m, s].
_SWEEP = 0.6
_PERIOD = 8.0

_CAMERA = {
    "lookat": (0.3, 0.0, 0.4),
    "distance": 2.4,
    "azimuth": 180.0,
    "elevation": -20.0,
}

_COLORS = {
    "task": (0.2, 0.75, 0.7, 1.0),
    "exact": (0.35, 0.45, 1.0, 1.0),
    "yielding": (1.0, 0.6, 0.1, 1.0),
    "failed": (0.9, 0.2, 0.25, 1.0),
}


@dataclass
class Arm:
    mode: str
    offset: np.ndarray
    configuration: mink.Configuration
    ee_task: mink.FrameTask
    tasks: list[mink.Task]
    constraints: list[mink.Task | mink.Elastic]
    limits: list[mink.Limit]
    qpos: slice
    site: str
    status: str = "exact"


def build_scene() -> mujoco.MjModel:
    """Panda scene with one extra arm on each side and a small floor."""
    spec = mujoco.MjSpec.from_file(_SCENE.as_posix())
    spec.delete(spec.body("target"))
    for i, mode in enumerate(_MODES):
        if i != 1:
            frame = spec.worldbody.add_frame(pos=[0.0, (i - 1) * _SPACING, 0.0])
            arm = mujoco.MjSpec.from_file(_ARM.as_posix())
            frame.attach_body(arm.body("link0"), f"{mode}/", "")

    spec.delete(spec.geom("floor"))
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0.35, 0.0, 0.0],
        size=[0.75, 1.35, 0.05],
        material="groundplane",
    )

    spec.visual.quality.offsamples = 8
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    return spec.compile()


def build_arm(scene: mujoco.MjModel, i: int, mode: str) -> Arm:
    model = mujoco.MjModel.from_xml_path(_ARM.as_posix())
    configuration = mink.Configuration(model)
    configuration.update_from_keyframe("home")

    ee_task = mink.FrameTask("attachment_site", "site", 1.0, 1.0)
    posture_task = mink.PostureTask(model, cost=_POSTURE_COST)
    posture_task.set_target_from_configuration(configuration)
    tasks: list[mink.Task] = [posture_task]
    constraints: list[mink.Task | mink.Elastic] = []
    if mode == "task":
        tasks.append(ee_task)
    elif mode == "hard":
        constraints.append(ee_task)
    else:
        constraints.append(mink.Elastic(ee_task, penalty=_PENALTY))

    limits = [
        mink.ConfigurationLimit(model),
        mink.VelocityLimit(model, {f"joint{j}": _MAX_VELOCITY for j in range(1, 8)}),
    ]

    prefix = "" if i == 1 else f"{mode}/"
    start = scene.jnt_qposadr[scene.joint(f"{prefix}joint1").id]
    return Arm(
        mode=mode,
        offset=np.array([0.0, (i - 1) * _SPACING, 0.0]),
        configuration=configuration,
        ee_task=ee_task,
        tasks=tasks,
        constraints=constraints,
        limits=limits,
        qpos=slice(start, start + model.nq),
        site=f"{prefix}attachment_site",
    )


def step(arm: Arm, target: mink.SE3, dt: float) -> None:
    """Solve one IK step toward the target, which is given in the arm's base frame."""
    arm.ee_task.set_target(target)
    configuration = arm.configuration
    try:
        v = mink.solve_ik(
            configuration,
            arm.tasks,
            dt,
            "daqp",
            limits=arm.limits,
            constraints=arm.constraints,
        )
    except mink.NoSolutionFound:
        arm.status = "failed"
        return
    if arm.mode == "task":
        arm.status = "task"
    else:
        J = arm.ee_task.compute_jacobian(configuration)
        residual = J @ v * dt + arm.ee_task.compute_error(configuration)
        arm.status = "exact" if np.abs(residual).max() < 1e-6 else "yielding"
    configuration.integrate_inplace(v, dt)


def target_at(home: mink.SE3, t: float) -> mink.SE3:
    """Home pose pushed forward by a smooth out-and-back sweep."""
    s = 0.5 * _SWEEP * (1.0 - np.cos(2.0 * np.pi * t / _PERIOD))
    return mink.SE3.from_translation(np.array([s, 0.0, 0.0])) @ home


def draw(
    scn: mujoco.MjvScene, arms: list[Arm], data: mujoco.MjData, target: mink.SE3
) -> None:
    """Append each arm's goal and a line from its end effector to the goal."""
    for arm in arms:
        rgba = np.array(_COLORS[arm.status], dtype=np.float32)
        goal = target.translation() + arm.offset
        sphere, line = scn.geoms[scn.ngeom], scn.geoms[scn.ngeom + 1]
        mujoco.mjv_initGeom(
            sphere,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.035, 0.0, 0.0]),
            goal,
            np.eye(3).ravel(),
            rgba,
        )
        mujoco.mjv_initGeom(
            line,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).ravel(),
            rgba,
        )
        mujoco.mjv_connector(
            line, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012, data.site(arm.site).xpos, goal
        )
        scn.ngeom += 2


def main() -> None:
    model = build_scene()
    data = mujoco.MjData(model)
    arms = [build_arm(model, i, mode) for i, mode in enumerate(_MODES)]
    home = arms[0].configuration.get_transform_frame_to_world("attachment_site", "site")

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        viewer.cam.lookat[:] = _CAMERA["lookat"]
        viewer.cam.distance = _CAMERA["distance"]
        viewer.cam.azimuth = _CAMERA["azimuth"]
        viewer.cam.elevation = _CAMERA["elevation"]

        t = 0.0
        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            target = target_at(home, t)
            for arm in arms:
                step(arm, target, rate.dt)
                data.qpos[arm.qpos] = arm.configuration.q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_camlight(model, data)
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                draw(viewer.user_scn, arms, data, target)
            viewer.sync()
            rate.sleep()
            t += rate.dt


if __name__ == "__main__":
    main()
