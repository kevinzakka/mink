"""Pour from a bottle into a cup with two ALOHA arms.

The left gripper holds a cup and a ``FrameTask`` places it at the mocap target. The
right gripper holds a bottle. A ``RelativeFrameTask`` places the bottle relative to the
left gripper, tilted with its lip over the cup rim. Drag the cup target: the bottle
follows and keeps pouring.

    uv run mjpython examples/arm_aloha_pour.py
"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "aloha" / "scene.xml"
_MUG = (_HERE / "mug" / "mug.obj").resolve()

_MUG_SCALE = 0.55  # Espresso-cup size, so it fits the jaws.
_MUG_H, _MUG_R = 0.072 * _MUG_SCALE, 0.045 * _MUG_SCALE  # Cup height and radius [m].
_BOTTLE_H, _BOTTLE_R = 0.24, 0.024  # Bottle height and body radius [m].
_BOTTLE_GRIP = 0.08  # Grasp height on the bottle body [m].
_CUP = np.array([-0.06, -0.02, 0.22])  # Cup target position [m].
_TILT = -120.0  # Bottle roll about the right gripper's approach axis [deg].


def _bottle_mesh(n: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Revolve a bottle profile (r, z) about z. Origin at the base."""
    r = _BOTTLE_R
    t = np.linspace(0.1, np.pi / 2, 8)
    prof = [(1e-4, 0.0), (r - 0.002, 0.0), (r, 0.004), (r, 0.14)]
    prof += list(zip(r - (r - 0.011) * (1 - np.cos(t)), 0.14 + 0.05 * np.sin(t)))
    prof += [(0.011, 0.2), (0.011, 0.228), (0.0125, 0.23), (0.0125, _BOTTLE_H)]
    prof += [(1e-4, _BOTTLE_H)]
    a = 2 * np.pi * np.arange(n) / n
    verts = [[pr * np.cos(ak), pr * np.sin(ak), pz] for pr, pz in prof for ak in a]
    faces = []
    for i in range(len(prof) - 1):
        for k in range(n):
            p, q = i * n + k, i * n + (k + 1) % n
            faces += [[p, q, q + n], [p, q + n, p + n]]
    return np.array(verts), np.array(faces)


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(_XML.as_posix())
    spec.delete(spec.body("right/target"))  # The bottle is placed relative to the cup.
    spec.add_mesh(name="mug", file=_MUG.as_posix(), scale=[0.01 * _MUG_SCALE] * 3)
    v, f = _bottle_mesh()
    spec.add_mesh(
        name="bottle", uservert=v.ravel().tolist(), userface=f.ravel().tolist()
    )

    # Cup: held at mid height, handle facing away. The mesh is +y up.
    cup_pos = spec.site("left/gripper").pos - np.array([0.0, 0.0, _MUG_H / 2])
    spec.body("left/gripper_link").add_geom(
        name="cup",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="mug",
        pos=cup_pos.tolist(),
        quat=[0, 0, 1, 1],
        rgba=[0.95, 0.95, 0.94, 1],
        contype=0,
        conaffinity=0,
        group=2,
    )
    # Bottle: held around its body, its axis along the gripper z.
    bottle_pos = spec.site("right/gripper").pos - np.array([0.0, 0.0, _BOTTLE_GRIP])
    spec.body("right/gripper_link").add_geom(
        name="bottle",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="bottle",
        pos=bottle_pos.tolist(),
        rgba=[0.35, 0.34, 0.84, 1],
        contype=0,
        conaffinity=0,
        group=2,
    )
    return spec.compile()


if __name__ == "__main__":
    model = build_model()
    configuration = mink.Configuration(model)
    configuration.update_from_keyframe("neutral_pose")

    # Close each jaw on its object.
    q = configuration.q
    for side, radius in (("left", _MUG_R), ("right", _BOTTLE_R)):
        for finger in ("left_finger", "right_finger"):
            q[model.joint(f"{side}/{finger}").qposadr[0]] = 0.015 + radius - 0.00075
    configuration.update(q)

    cup_task = mink.FrameTask("left/gripper", "site", 1.0, 1.0, lm_damping=1.0)
    pour_task = mink.RelativeFrameTask(
        "right/gripper", "site", "left/gripper", "site", 1.0, 1.0, lm_damping=1.0
    )
    posture_task = mink.PostureTask(model, cost=1e-4)
    posture_task.set_target_from_configuration(configuration)
    tasks = [cup_task, pour_task, posture_task]
    limits = [mink.ConfigurationLimit(model)]

    # Cup target: keep the left gripper's orientation, move it to _CUP.
    T_l = configuration.get_transform_frame_to_world("left/gripper", "site")
    T_r = configuration.get_transform_frame_to_world("right/gripper", "site")
    T_cup = mink.SE3.from_rotation_and_translation(T_l.rotation(), _CUP)

    # Bottle target: roll the right gripper about its approach axis (x), then place
    # the bottle so that its lip sits just above the cup rim.
    R_r = T_r.rotation().as_matrix()
    tilt = mink.SO3.exp(np.deg2rad(_TILT) * R_r[:, 0])
    axis = tilt.as_matrix() @ R_r[:, 2]
    lip = _CUP + np.array([0.0, 0.0, _MUG_H / 2 + 0.015])
    grip = lip - (_BOTTLE_H - _BOTTLE_GRIP) * axis
    T_bottle = mink.SE3.from_rotation_and_translation(tilt @ T_r.rotation(), grip)

    # Solve the pour before opening the viewer, so the scene starts in it.
    cup_task.set_target(T_cup)
    pour_task.set_target(T_cup.inverse() @ T_bottle)
    for _ in range(600):
        vel = mink.solve_ik(
            configuration, tasks, 0.005, "daqp", damping=1e-3, limits=limits
        )
        configuration.integrate_inplace(vel, 0.005)
    posture_task.set_target_from_configuration(configuration)

    model, data = configuration.model, configuration.data
    mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            cup_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
            vel = mink.solve_ik(
                configuration, tasks, rate.dt, "daqp", damping=1e-3, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)
            viewer.sync()
            rate.sleep()
