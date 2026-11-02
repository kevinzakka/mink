"""Whole-body reach: a Unitree H2 with Sharpa hands takes a book from a low shelf.

One solve moves the floating base, legs, waist, head, arms and fingers together:

* A ``LookAtTask`` points the head at the book.
* A ``FrameTask`` places the right hand so that its index fingertip rests on the
  book's top edge, with the palm toward the spine.
* A ``FrameTask`` on a site at the center of the closed left fist braces the hand
  around a shelf post.
* A ``FrameTask`` on each foot keeps it planted, and a ``ComTask`` keeps the center
  of mass over the support polygon.
* A ``PostureTask`` prefers bent knees, and holds a closed left fist and a relaxed
  right hand.

Every target is a mocap body. Drag them in the viewer to move the robot.

    uv run mjpython examples/humanoid_h2_bookshelf.py
"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "unitree_h2" / "scene.xml"

_FINGERS = ("index", "middle", "ring", "pinky")
_SHELF = (0.75, 0.0, 0.36, 0.15)  # Center x, center y, half width, half depth [m].
_BOARDS = (0.02, 0.45, 0.85, 1.25, 1.65)  # Board heights [m].
_POST_Z = 0.95  # Height of the left hand on the post [m].
_STANCE = 0.08  # Each foot moves out this far [m].
_SQUAT = {"hip_pitch_joint": -0.8, "knee_joint": 1.3, "ankle_pitch_joint": -0.5}
_FIST = 0.7  # Closure of the left fist around the post.
_INDIGO = (0.35, 0.34, 0.84, 1.0)
_GRAYS = ((0.56, 0.57, 0.6, 1), (0.68, 0.69, 0.72, 1), (0.44, 0.45, 0.48, 1))
_WOOD = (0.86, 0.85, 0.83, 1.0)
_MARKER = (0.6, 0.3, 0.3, 0.2)


def _hand_postures() -> dict[str, float]:
    """Left hand closed into a fist, right hand relaxed with more curl toward the pinky."""
    q = {}
    for f in _FINGERS:
        for j, v in (("MCP_FE", 1.0), ("PIP", 1.4), ("DIP", 1.0)):
            q[f"left_{f}_{j}"] = _FIST * v
    q.update(
        left_thumb_CMC_FE=_FIST * 1.3,
        left_thumb_MCP_FE=_FIST * 0.7,
        left_thumb_IP=_FIST * 0.8,
    )
    for f, curl in zip(_FINGERS, (0.3, 0.6, 0.8, 1.0)):
        for j, v in (("MCP_FE", curl), ("PIP", 1.2 * curl), ("DIP", 0.8 * curl)):
            q[f"right_{f}_{j}"] = v
    q.update(right_thumb_CMC_FE=0.5, right_thumb_MCP_FE=0.3, right_thumb_IP=0.3)
    return q


def _box(body, size, pos, rgba) -> None:
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=size,
        pos=pos,
        rgba=rgba,
        contype=0,
        conaffinity=0,
    )


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(_XML.as_posix())

    # Gaze site between the eyes, x forward.
    spec.body("head_yaw_link").add_site(name="gaze", pos=[0.09, 0.0, 0.06])
    # Center of the closed left fist: x along the finger row, z from palm to fingers.
    spec.body("left_hand_C_MC").add_site(
        name="left_grip",
        pos=[0.04171, -0.01362, 0.09447],
        quat=[0.51519, 0.50068, 0.55203, 0.42327],
    )

    # Open shelf with four corner posts and three rows of books.
    world = spec.worldbody
    sx, sy, w, d = _SHELF
    for z in _BOARDS:
        _box(world, [d, w, 0.012], [sx, sy, z], _WOOD)
    for i in (-1, 1):
        for k in (-1, 1):
            pos = [sx + i * (d - 0.012), sy + k * (w - 0.012), 0.83]
            _box(world, [0.012, 0.012, 0.83], pos, _WOOD)
    for level in (1, 2, 3):
        z0 = _BOARDS[level] + 0.012
        for i, y in enumerate(sy + np.linspace(-0.3, 0.3, 11)):
            if level != 3 and (i + level) % 4 == 0:
                continue  # Gaps on the lower shelves.
            h = 0.1 + 0.015 * ((i * 7 + level) % 3)
            rgba = _GRAYS[(i + level) % 3]
            if (level, i) == (2, 1):  # The book to take.
                h, rgba = 0.11, _INDIGO
            _box(world, [0.08, 0.02, h], [sx - 0.04, y, z0 + h], rgba)

    # Draggable targets.
    sphere, box = mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_BOX
    for name, type_, size in (
        ("gaze_target", sphere, 0.03),
        ("hand_target", box, 0.03),
        ("grip_target", box, 0.03),
        ("left_foot_target", box, 0.04),
        ("right_foot_target", box, 0.04),
    ):
        body = world.add_body(name=name, mocap=True)
        body.add_geom(
            type=type_,
            size=[size] * 3,
            rgba=_MARKER,
            contype=0,
            conaffinity=0,
        )
    return spec.compile()


def _set_mocap(data, mid, T: mink.SE3) -> None:
    data.mocap_pos[mid] = T.translation()
    data.mocap_quat[mid] = T.rotation().wxyz


if __name__ == "__main__":
    model = build_model()
    configuration = mink.Configuration(model)
    data = configuration.data

    # Posture: a crouch on bent knees, a closed left fist and a relaxed right hand.
    cost = np.full(model.nv, 0.1)
    cost[:6] = 0.0
    q_ref = configuration.q.copy()
    for name, v in _hand_postures().items():
        j = model.joint(name)
        q_ref[j.qposadr[0]] = np.clip(v, *model.jnt_range[j.id])
        cost[j.dofadr[0]] = 1.0
    for side in ("left", "right"):
        for name, v in _SQUAT.items():
            j = model.joint(f"{side}_{name}")
            q_ref[j.qposadr[0]], cost[j.dofadr[0]] = v, 1.0
    posture_task = mink.PostureTask(model, cost=cost)
    posture_task.set_target(q_ref)

    feet = ("left_ankle_pitch_link", "right_ankle_pitch_link")
    feet_tasks = [mink.FrameTask(f, "body", 50.0, 10.0, lm_damping=1.0) for f in feet]
    com_task = mink.ComTask(cost=30.0)
    gaze_task = mink.LookAtTask(
        "gaze", "site", axis=np.array([1.0, 0.0, 0.0]), cost=1.0, lm_damping=1.0
    )
    hand_task = mink.FrameTask("right_hand_C_MC", "body", 50.0, 5.0, lm_damping=1.0)
    grip_task = mink.FrameTask("left_grip", "site", 20.0, 2.0, lm_damping=1.0)
    tasks = [
        gaze_task,
        hand_task,
        grip_task,
        com_task,
        *feet_tasks,
        posture_task,
    ]
    limits = [mink.ConfigurationLimit(model)]

    # Initial targets: the book, its top edge, a post, and a wider stance.
    sx, sy, w, d = _SHELF
    book = np.array([sx - 0.04, sy - 0.24, _BOARDS[2] + 0.122])
    mids = {n: model.body(f"{n}_target").mocapid[0] for n in ("gaze", "hand", "grip")}
    _set_mocap(data, mids["gaze"], mink.SE3.from_translation(book))
    # Hand pose that rests the relaxed index fingertip on the book's top edge.
    edge = np.array([book[0] - 0.068, book[1], book[2] + 0.122])
    palm = mink.SO3.from_y_radians(np.deg2rad(20.0))
    relaxed = mink.Configuration(model, q_ref)
    T_tip = relaxed.get_transform(
        "right_index_fingertip", "site", "right_hand_C_MC", "body"
    )
    hand = edge - palm.as_matrix() @ T_tip.translation()
    _set_mocap(data, mids["hand"], mink.SE3.from_rotation_and_translation(palm, hand))
    post = np.array([sx - d + 0.012, sy + w - 0.012, _POST_Z])
    R_grip = mink.SO3.from_matrix(np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], float))
    _set_mocap(data, mids["grip"], mink.SE3.from_rotation_and_translation(R_grip, post))
    feet_mid = np.zeros(3)
    for f, side, sign in zip(feet, ("left", "right"), (1.0, -1.0)):
        T = configuration.get_transform_frame_to_world(f, "body")
        feet_mid += 0.5 * T.translation()
        T = mink.SE3.from_translation(np.array([0.0, sign * _STANCE, 0.0])) @ T
        _set_mocap(data, model.body(f"{side}_foot_target").mocapid[0], T)
    # Center of mass over the feet, and lower: the hips sink into a crouch.
    com_offset = data.subtree_com[1] - feet_mid - np.array([0.0, 0.0, 0.22])

    def mocap(name: str) -> mink.SE3:
        return mink.SE3.from_mocap_name(model, data, f"{name}_target")

    def step(dt: float) -> None:
        gaze_task.set_target(mocap("gaze").translation())
        hand_task.set_target(mocap("hand"))
        grip_task.set_target(mocap("grip"))
        for task, side in zip(feet_tasks, ("left", "right")):
            task.set_target(mocap(f"{side}_foot"))
        feet_mid = sum(mocap(f"{s}_foot").translation() for s in ("left", "right")) / 2
        com_task.set_target(feet_mid + com_offset)
        vel = mink.solve_ik(
            configuration, tasks, dt, "daqp", damping=1e-1, limits=limits
        )
        configuration.integrate_inplace(vel, dt)

    # Solve the reach before opening the viewer, so the scene starts in it.
    for _ in range(1000):
        step(0.01)

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        rate = RateLimiter(frequency=100.0, warn=False)
        while viewer.is_running():
            step(rate.dt)
            viewer.sync()
            rate.sleep()
