"""Quadratic, hard, and elastic end-effector tasks on a Panda, compared numerically.

A posture task pulls toward the home keyframe while the end effector tracks a target
away from home. The end-effector task is passed in one of three ways:

- l2: as a quadratic cost in ``tasks``. Tracking is biased by the posture task.
- hard: as an equality in ``constraints``. Exact, but infeasible when limits bind.
- elastic: wrapped in :class:`mink.Elastic` in ``constraints``. Exact while the
  constraint force stays below the penalty, and yields otherwise.

    uv run python examples/arm_panda_elastic.py
"""

from pathlib import Path

import mujoco
import numpy as np
import numpy.typing as npt
import qpsolvers

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "franka_emika_panda" / "mjx_scene.xml"
_DT = 0.01
_SOLVER = "daqp"


def setup(model, offset):
    configuration = mink.Configuration(model)
    configuration.update_from_keyframe("home")
    ee_task = mink.FrameTask("attachment_site", "site", 1.0, 1.0)
    home = configuration.get_transform_frame_to_world("attachment_site", "site")
    ee_task.set_target(home @ mink.SE3.from_translation(np.asarray(offset)))
    posture_task = mink.PostureTask(model, cost=1e-1)
    posture_task.set_target_from_configuration(configuration)
    return configuration, ee_task, posture_task


def split(ee_task, posture_task, mode, penalty: npt.ArrayLike = 1e3):
    if mode == "l2":
        return [ee_task, posture_task], None
    if mode == "hard":
        return [posture_task], [ee_task]
    return [posture_task], [mink.Elastic(ee_task, penalty=penalty)]


def track(model, offset, mode, limits, penalty: npt.ArrayLike = 1e3, steps=500):
    configuration, ee_task, posture_task = setup(model, offset)
    tasks, constraints = split(ee_task, posture_task, mode, penalty)
    for step in range(steps):
        try:
            v = mink.solve_ik(
                configuration,
                tasks,
                _DT,
                _SOLVER,
                limits=limits(model),
                constraints=constraints,
            )
        except mink.NoSolutionFound:
            return f"NoSolutionFound at step {step}"
        configuration.integrate_inplace(v, _DT)
    e = ee_task.compute_error(configuration)
    e_posture = posture_task.compute_error(configuration)
    return (
        f"|pos err| {np.linalg.norm(e[:3]):.2e}  |rot err| {np.linalg.norm(e[3:]):.2e}"
        f"  |posture err| {np.linalg.norm(e_posture):.3f}"
    )


def position_limits(model):
    return [mink.ConfigurationLimit(model)]


def rate_limits(model):
    velocities = {model.joint(i).name: np.pi for i in range(model.njnt)}
    return [mink.ConfigurationLimit(model), mink.VelocityLimit(model, velocities)]


def main():
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    reachable = [0.15, -0.1, 0.1]
    unreachable = [1.5, 0.0, 0.0]

    print("1. Reachable target, joint position limits only.")
    for mode in ("l2", "hard", "elastic"):
        print(f"   {mode:8s} {track(model, reachable, mode, position_limits)}")

    print("\n2. Same target, velocity limit pi rad/s.")
    for mode in ("l2", "hard", "elastic"):
        print(f"   {mode:8s} {track(model, reachable, mode, rate_limits)}")

    print("\n3. Unreachable target, velocity limit pi rad/s.")
    for mode in ("l2", "hard", "elastic"):
        print(f"   {mode:8s} {track(model, unreachable, mode, rate_limits)}")

    print("\n4. Zero penalty frees a component: hold position, release orientation.")
    for name, penalty in (("6-D", 1e3), ("position", [1e3] * 3 + [0.0] * 3)):
        result = track(model, reachable, "elastic", position_limits, penalty)
        print(f"   {name:8s} {result}")

    print(
        "\n5. One QP: exact once rho exceeds every |lambda_i| of the hard constraint."
    )
    configuration, ee_task, posture_task = setup(model, reachable)
    problem = mink.build_ik(
        configuration, [posture_task], _DT, limits=[], constraints=[ee_task]
    )
    result = qpsolvers.solve_problem(problem, solver=_SOLVER)
    assert result.x is not None and result.y is not None
    lam = np.abs(result.y)
    print(f"   hard multipliers |lambda| = {np.array2string(lam, precision=4)}")
    jacobian = ee_task.compute_jacobian(configuration)
    error = ee_task.compute_error(configuration)
    for scale in (0.5, 0.9, 1.1, 2.0):
        penalty = scale * lam.max()
        v = mink.solve_ik(
            configuration,
            [posture_task],
            _DT,
            _SOLVER,
            limits=[],
            constraints=[mink.Elastic(ee_task, penalty=penalty)],
        )
        residual = jacobian @ v * _DT + error
        held = "".join("x" if abs(r) < 1e-7 else "." for r in residual)
        gap = np.linalg.norm(v * _DT - result.x)
        print(
            f"   rho = {scale:.1f} max|lambda|: held [{held}]  |dq - dq_hard| {gap:.1e}"
        )


if __name__ == "__main__":
    main()
