"""Engrave the MuJoCo "M" into a domed block, keeping the tool normal to the surface.

A Panda traces a precomputed toolpath over a dome. A ``FrameTask`` drives the tool
tip along the path while :class:`mink.AxisAlignTask` keeps the spindle aligned with
the local surface normal, leaving rotation about the tool free. Material is removed
live by carving the block's heightfield under the bit.

``AxisAlignTask`` is the sibling of :class:`mink.LookAtTask`: look-at points a frame
axis at a point, axis-align points it along a direction (here, the surface normal).
When ``--no-level`` is provided, the task is dropped and the tool no longer tracks
the normal, cutting the dome at the wrong angle.

    uv run mjpython examples/arm_panda_engrave.py
    uv run mjpython examples/arm_panda_engrave.py --no-level
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
_SCENE = _HERE / "franka_emika_panda" / "scene.xml"
_LOGO = _HERE / "mujoco_M.png"
_TOOLPATH = _HERE / "mujoco_M_toolpath.npy"

_JOINTS = [f"joint{i}" for i in range(1, 8)]
_FLANGE_Z = 0.107  # Flange offset along link7's +z, where the hand used to mount.
_TOOL_LEN = 0.14  # Tool length past the flange.

# Domed block: a shallow cap of a sphere of radius ``_DOME_R`` centered below the
# peak. The gentle curvature keeps the toolpath dexterous while making the
# axis-align task do visible work (the tool tilts up to ~20 deg across the dome).
_BLOCK_XY = np.array([0.5, 0.0])
_DOME_PEAK = 0.17
_DOME_R = 0.26
_SPHERE_C = np.array([0.0, 0.0, _DOME_PEAK - _DOME_R])  # block-local sphere center
_ENGRAVE = 0.02  # Engraving depth.
_HALF = 0.13  # Block half-width.
_PATCH = 0.10  # Half-extent the glyph is fit into.
_GRID = 240  # Heightfield resolution.
_BIT_R = 0.0065  # Visible cutting-tip radius.
_CARVE_R = 0.014  # Material-removal radius (clears the M's acute corners).
_SIGMA = 6.0  # Joint-trajectory smoothing (frames).

_xs = np.linspace(-_HALF, _HALF, _GRID)
_XX, _YY = np.meshgrid(_xs, _xs)
_DOME = (
    _DOME_PEAK - _DOME_R + np.sqrt(np.clip(_DOME_R**2 - _XX**2 - _YY**2, 1e-6, None))
)
_ZBASE = _DOME.min() - 0.001
_ZTOP = _DOME.max()


# --- The surface: a dome, and the M engraved into it. ---------------------------


def _resize(a: np.ndarray, n: int) -> np.ndarray:
    r = np.clip((np.arange(n) * a.shape[0] / n).astype(int), 0, a.shape[0] - 1)
    c = np.clip((np.arange(n) * a.shape[1] / n).astype(int), 0, a.shape[1] - 1)
    return a[r][:, c]


def _glyph_mask() -> np.ndarray:
    """Binary MuJoCo-M on the heightfield grid (row=y, col=x), fit into the patch."""
    img = iio.imread(_LOGO).astype(float) / 255.0
    if img.ndim == 3:
        img = img[..., 0]
    inset = int(_GRID * (1 - _PATCH / _HALF) / 2)
    inner = _GRID - 2 * inset
    grid = np.zeros((_GRID, _GRID))
    grid[inset : inset + inner, inset : inset + inner] = _resize(img, inner) > 0.5
    return np.rot90((grid > 0.5).astype(np.uint8))


_MASK_BIN = _glyph_mask()
# Soft mask drives the cut depth (eased edges so the tool plunges smoothly).
_MASK = _MASK_BIN.astype(float)
for _ in range(6):
    _MASK = (
        _MASK
        + np.roll(_MASK, 1, 0)
        + np.roll(_MASK, -1, 0)
        + np.roll(_MASK, 1, 1)
        + np.roll(_MASK, -1, 1)
    ) / 5
_TARGET = (_DOME - _ENGRAVE * _MASK - _ZBASE) / (_ZTOP - _ZBASE)


def _mask_at(lx: float, ly: float) -> float:
    fx = (lx + _HALF) / (2 * _HALF) * (_GRID - 1)
    fy = (ly + _HALF) / (2 * _HALF) * (_GRID - 1)
    j = int(np.clip(np.floor(fx), 0, _GRID - 2))
    i = int(np.clip(np.floor(fy), 0, _GRID - 2))
    a, b = fx - j, fy - i
    return float(
        _MASK[i, j] * (1 - a) * (1 - b)
        + _MASK[i, j + 1] * a * (1 - b)
        + _MASK[i + 1, j] * (1 - a) * b
        + _MASK[i + 1, j + 1] * a * b
    )


def _dome_z(lx: float, ly: float) -> float:
    return _DOME_PEAK - _DOME_R + np.sqrt(max(_DOME_R**2 - lx * lx - ly * ly, 1e-6))


def _dome_normal(lx: float, ly: float) -> np.ndarray:
    n = np.array([lx, ly, _dome_z(lx, ly)]) - _SPHERE_C
    return n / np.linalg.norm(n)


# The toolpath: waypoints ``(x, y, lift)`` in block-local meters, where ``lift`` is
# 0 at cut depth and positive while traversing above disjoint regions. Precomputed
# by contour-parallel pocketing of the glyph and vendored alongside this example.
_PATH = np.load(_TOOLPATH)
_SEG = np.linalg.norm(np.diff(_PATH[:, :2], axis=0), axis=1) + np.abs(
    np.diff(_PATH[:, 2])
)
_CUM = np.concatenate([[0], np.cumsum(_SEG)])


def _path_at(s: float) -> tuple[float, float, float]:
    t = s * _CUM[-1]
    i = int(np.clip(np.searchsorted(_CUM, t) - 1, 0, len(_PATH) - 2))
    f = (t - _CUM[i]) / max(_SEG[i], 1e-9)
    p = _PATH[i] * (1 - f) + _PATH[i + 1] * f
    return p[0], p[1], p[2]


def build_model() -> mujoco.MjModel:
    """No-hand Panda with an end mill on link7 and a domed block to engrave."""
    spec = mujoco.MjSpec.from_file(_SCENE.as_posix())

    # Drop the gripper (the mill is the tool).
    spec.delete(spec.body("hand"))

    # Tweak lighting.
    spec.visual.headlight.ambient[:] = [0.3, 0.3, 0.3]
    spec.visual.headlight.diffuse[:] = [0.4, 0.4, 0.4]
    spec.visual.headlight.specular[:] = [0, 0, 0]
    spec.visual.quality.offsamples = 8
    for light in spec.lights:
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        light.diffuse = [0.3, 0.3, 0.32]
        light.castshadow = False
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.delete(spec.geom("floor"))
    for tex in spec.textures:
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
            tex.rgb1, tex.rgb2 = [0.14, 0.15, 0.18], [0.03, 0.03, 0.05]
    spec.visual.rgba.haze[:] = [0.05, 0.05, 0.07, 1.0]
    spec.worldbody.add_light(
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        dir=[0.3, 0.5, -0.9],
        diffuse=[0.6, 0.58, 0.54],
        specular=[0.4, 0.4, 0.4],
        castshadow=False,
    )

    # Add mill tooltip.
    link7 = spec.body("link7")
    link7.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[0, 0, _FLANGE_Z, 0, 0, _FLANGE_Z + _TOOL_LEN - 0.035],
        size=[0.0095],
        rgba=[0.55, 0.57, 0.62, 1],
    )
    link7.add_geom(  # cutting tip; recolored hot while it is removing material
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[0, 0, _FLANGE_Z + _TOOL_LEN - 0.035, 0, 0, _FLANGE_Z + _TOOL_LEN],
        size=[_BIT_R],
        rgba=[0.3, 0.3, 0.34, 1],
        name="bit",
    )
    link7.add_site(
        name="tip", pos=[0, 0, _FLANGE_Z + _TOOL_LEN], size=[0.003] * 3, group=4
    )

    spec.add_material(
        name="block", rgba=[0.12, 0.28, 0.66, 1], specular=0.35, shininess=0.45
    )
    spec.add_hfield(
        name="block",
        nrow=_GRID,
        ncol=_GRID,
        size=[_HALF, _HALF, _ZTOP - _ZBASE, 0.08],
        userdata=((_DOME - _ZBASE) / (_ZTOP - _ZBASE)).ravel().tolist(),
    )
    body = spec.worldbody.add_body(
        name="block", pos=[_BLOCK_XY[0], _BLOCK_XY[1], _ZBASE]
    )
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="block", material="block"
    )

    return spec.compile()


def carve(model, field, a, b):
    """Remove material along the swept segment a->b (local block coords)."""
    ab = b - a
    L2 = ab @ ab + 1e-12
    t = np.clip(((_XX - a[0]) * ab[0] + (_YY - a[1]) * ab[1]) / L2, 0, 1)
    dd = (_XX - (a[0] + t * ab[0])) ** 2 + (_YY - (a[1] + t * ab[1])) ** 2
    np.minimum(field, np.where(dd < _CARVE_R**2, _TARGET, 1e9), out=field)
    model.hfield_data[:] = np.clip(field, 0, 1).ravel()


def _tool_state(model, data, site_id):
    """Tip (block-local), whether engaged in material, and recolor the bit."""
    tip = data.site_xpos[site_id]
    local = np.array([tip[0] - _BLOCK_XY[0], tip[1] - _BLOCK_XY[1]])
    cutting = (_dome_z(*local) - tip[2]) > 0.3 * _ENGRAVE
    model.geom_rgba[model.geom("bit").id] = (
        [1.0, 0.42, 0.08, 1] if cutting else [0.3, 0.3, 0.34, 1]
    )
    return local, cutting


def _flat_field() -> np.ndarray:
    return (_DOME - _ZBASE) / (_ZTOP - _ZBASE)


def plan(model: mujoco.MjModel, level: bool) -> np.ndarray:
    """Solve IK along the toolpath, then low-pass filter the joint trajectory.

    Returns the joint trajectory to play back. The Gaussian smoothing in joint
    space removes the residual vibration left by the per-step IK while keeping the
    tool within a fraction of a millimeter of the surface.
    """
    configuration = mink.Configuration(model)
    configuration.update_from_keyframe("home")

    frame_task = mink.FrameTask(
        "tip", "site", position_cost=1.0, orientation_cost=0.0, lm_damping=1e-2
    )
    align_task = mink.AxisAlignTask("tip", "site", (0, 0, 1), cost=1.0, lm_damping=1e-3)
    posture_task = mink.PostureTask(model, cost=1e-2)
    posture_task.set_target_from_configuration(configuration)
    tasks = [frame_task, posture_task]
    if level:  # without it, only the tip position is tracked and the tool digs in
        tasks.insert(1, align_task)
    limits = [
        mink.ConfigurationLimit(model),
        mink.VelocityLimit(model, dict.fromkeys(_JOINTS, np.pi)),
    ]

    def seek(s: float) -> None:
        """Aim the tip at the toolpath point at fraction ``s`` along the path."""
        lx, ly, lift = _path_at(s)
        z = _dome_z(lx, ly) - _ENGRAVE * _mask_at(lx, ly) + lift
        frame_task.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3.identity(), np.array([_BLOCK_XY[0] + lx, _BLOCK_XY[1] + ly, z])
            )
        )
        align_task.set_target(-_dome_normal(lx, ly))  # link7 +z points into the surface

    seek(0.0)
    for _ in range(300):  # settle onto the start (lifted above the first point)
        vel = mink.solve_ik(configuration, tasks, 1 / 200, "daqp", 1e-2, limits=limits)
        configuration.integrate_inplace(vel, 1 / 200)

    frames = int(np.clip(_CUM[-1] / 0.0022, 600, 1300))
    qraw = np.zeros((frames, model.nq))
    for k in range(frames):
        seek(k / (frames - 1))
        for _ in range(6):
            vel = mink.solve_ik(
                configuration, tasks, 1 / 200, "daqp", 1e-2, limits=limits
            )
            configuration.integrate_inplace(vel, 1 / 200)
        qraw[k] = configuration.q

    home = model.key_qpos[model.key("home").id].copy()
    qraw = np.vstack(
        [np.linspace(home, qraw[0], 70), qraw, np.linspace(qraw[-1], home, 70)]
    )

    r = int(4 * _SIGMA)
    ker = np.exp(-(np.arange(-r, r + 1) ** 2) / (2 * _SIGMA**2))
    ker /= ker.sum()
    return np.stack(
        [
            np.convolve(np.pad(qraw[:, j], (r, r), "edge"), ker, "valid")
            for j in range(model.nq)
        ],
        axis=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-level",
        action="store_true",
        help="Drop the axis-align task; the tool no longer tracks the surface normal.",
    )
    args = parser.parse_args()

    model = build_model()
    qpath = plan(model, level=not args.no_level)

    data = mujoco.MjData(model)
    data.qpos[:] = qpath[0]
    mujoco.mj_forward(model, data)
    site_id = model.site("tip").id
    field = _flat_field()

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.cam.lookat[:] = [_BLOCK_XY[0], _BLOCK_XY[1], _DOME_PEAK]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.85, 195, -35
        rate = RateLimiter(frequency=60.0, warn=False)
        k, prev, dirty = 0, None, False
        while viewer.is_running():
            if np.allclose(data.qpos, model.qpos0):  # reset pressed: re-arm the stock
                field = _flat_field()
                model.hfield_data[:] = field.ravel()
                viewer.update_hfield(0)
                k, prev = 0, None
            data.qpos[:] = qpath[min(k, len(qpath) - 1)]
            mujoco.mj_forward(model, data)
            local, cutting = _tool_state(model, data, site_id)
            if cutting and prev is not None:
                carve(model, field, prev, local)
                dirty = True
            prev = local
            # Re-uploading the heightfield blocks the render thread, so throttle it.
            if dirty and k % 3 == 0:
                viewer.update_hfield(0)
                dirty = False
            viewer.sync()
            k = min(k + 1, len(qpath))
            rate.sleep()


if __name__ == "__main__":
    main()
