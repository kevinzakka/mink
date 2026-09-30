"""Three Pandas track one target: a quadratic task, a hard constraint, and an elastic
constraint.

All three arms get the same end-effector target and a posture task pulling toward
home. Double-click the target, then Ctrl + right-drag it, and compare:

- spring (left): the end effector is a quadratic task in ``tasks``. It settles short
  of the target, pulled back by the posture task.
- pin (middle): the end effector is a hard constraint in ``constraints``. It tracks
  exactly, and turns red with no solution when the target leaves the workspace or
  moves faster than the velocity limit allows.
- clutch (right): the end effector is wrapped in :class:`mink.Elastic`. It tracks
  exactly while it can, and turns orange and slips when it cannot.

The posture cost is exaggerated so the spring's offset is easy to see. Press Enter
to reset the arms, for example after the pin fails.

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

# (name, joint and site prefix in the scene, lateral offset of the base).
_ARMS = (("spring", "spring/", -1.2), ("pin", "", 0.0), ("clutch", "clutch/", 1.2))
_POSTURE_COST = 0.3
_MAX_VELOCITY = np.pi

# L1 penalties for (x, y, z, roll, pitch, yaw). With equal weights on meters and
# radians, a stretched arm can park where any position gain costs as much
# orientation error, so orientation gets the lower penalty and gives way first.
_PENALTY = np.array([1e3, 1e3, 1e3, 1e1, 1e1, 1e1])

_ENTER = 257

_COLORS = {
    "spring": (0.2, 0.75, 0.7, 1.0),
    "held": (0.35, 0.45, 1.0, 1.0),
    "slipping": (1.0, 0.6, 0.1, 1.0),
    "failed": (0.9, 0.2, 0.25, 1.0),
}


@dataclass
class Arm:
    name: str
    prefix: str
    offset: float
    configuration: mink.Configuration
    ee_task: mink.FrameTask
    tasks: list
    constraints: list | None
    limits: list
    qpos: slice
    status: str = "held"


def build_scene() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(_SCENE.as_posix())
    for _, prefix, offset in _ARMS:
        if prefix:
            frame = spec.worldbody.add_frame(pos=[0.0, offset, 0.0])
            arm = mujoco.MjSpec.from_file(_ARM.as_posix())
            frame.attach_body(arm.body("link0"), prefix, "")
    return spec.compile()


def build_arms(scene: mujoco.MjModel) -> list[Arm]:
    arms = []
    for name, prefix, offset in _ARMS:
        model = mujoco.MjModel.from_xml_path(_ARM.as_posix())
        configuration = mink.Configuration(model)
        configuration.update_from_keyframe("home")
        ee_task = mink.FrameTask("attachment_site", "site", 1.0, 1.0)
        posture_task = mink.PostureTask(model, cost=_POSTURE_COST)
        posture_task.set_target_from_configuration(configuration)
        if name == "spring":
            tasks, constraints = [ee_task, posture_task], None
        elif name == "pin":
            tasks, constraints = [posture_task], [ee_task]
        else:
            tasks, constraints = [posture_task], [mink.Elastic(ee_task, _PENALTY)]
        velocities = {f"joint{i}": _MAX_VELOCITY for i in range(1, 8)}
        limits = [
            mink.ConfigurationLimit(model),
            mink.VelocityLimit(model, velocities),
        ]
        start = scene.jnt_qposadr[scene.joint(f"{prefix}joint1").id]
        qpos = slice(start, start + model.nq)
        arms.append(
            Arm(
                name,
                prefix,
                offset,
                configuration,
                ee_task,
                tasks,
                constraints,
                limits,
                qpos,
            )
        )
    return arms


def reset(arms: list[Arm], model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for arm in arms:
        arm.configuration.update_from_keyframe("home")
        arm.status = "held"
        data.qpos[arm.qpos] = arm.configuration.q
    mujoco.mj_forward(model, data)
    mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")


def step(arm: Arm, dt: float) -> None:
    try:
        v = mink.solve_ik(
            arm.configuration,
            arm.tasks,
            dt,
            "daqp",
            limits=arm.limits,
            constraints=arm.constraints,
        )
    except mink.NoSolutionFound:
        arm.status = "failed"
        return
    residual = arm.ee_task.compute_jacobian(arm.configuration) @ v * dt
    residual += arm.ee_task.compute_error(arm.configuration)
    arm.configuration.integrate_inplace(v, dt)
    if arm.name == "spring":
        arm.status = "spring"
    else:
        arm.status = "held" if np.abs(residual).max() < 1e-6 else "slipping"


def describe(arm: Arm) -> str:
    error = 100.0 * np.linalg.norm(arm.ee_task.compute_error(arm.configuration)[:3])
    if arm.status == "spring":
        return f"{error:.1f} cm off"
    if arm.status == "failed":
        return "NO SOLUTION"
    if arm.status == "slipping":
        return f"slipping, {error:.1f} cm off"
    return "held exactly"


def draw(scn: mujoco.MjvScene, arms: list[Arm], data: mujoco.MjData, target) -> None:
    scn.ngeom = 0
    for arm in arms:
        rgba = np.array(_COLORS[arm.status], dtype=np.float32)
        goal = target.translation() + np.array([0.0, arm.offset, 0.0])
        tip = data.site(f"{arm.prefix}attachment_site").xpos
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.035, 0.0, 0.0]),
            goal,
            np.eye(3).ravel(),
            rgba,
        )
        scn.ngeom += 1
        mujoco.mjv_connector(
            scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012, tip, goal
        )
        scn.geoms[scn.ngeom].rgba[:] = rgba
        scn.ngeom += 1


if __name__ == "__main__":
    model = build_scene()
    data = mujoco.MjData(model)
    arms = build_arms(model)
    reset(arms, model, data)

    pressed = {"reset": False}

    def key_callback(key: int) -> None:
        if key == _ENTER:
            pressed["reset"] = True

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        viewer.cam.lookat[:] = (0.3, 0.0, 0.4)
        viewer.cam.distance = 3.6
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -20.0

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            if pressed["reset"]:
                pressed["reset"] = False
                reset(arms, model, data)
            target = mink.SE3.from_mocap_name(model, data, "target")
            for arm in arms:
                arm.ee_task.set_target(target)
                step(arm, rate.dt)
                data.qpos[arm.qpos] = arm.configuration.q
            mujoco.mj_kinematics(model, data)

            with viewer.lock():
                draw(viewer.user_scn, arms, data, target)
            viewer.set_texts(
                [
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "spring  (L2 task)\npin  (hard constraint)\nclutch  (mink.Elastic)",
                        "\n".join(describe(arm) for arm in arms),
                    ),
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                        "Double-click the target, then Ctrl + right-drag to move it."
                        " Enter resets.",
                        None,
                    ),
                ]
            )
            viewer.sync()
            rate.sleep()
