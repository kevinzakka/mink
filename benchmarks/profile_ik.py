"""Per-component IK profiler: attribute each step's wall time to its phases.

Where bench_ik.py answers "how fast is a step", this answers "where does the step
go". It mirrors the body of mink.solve_ik so each phase can be timed
independently:

  update          move task targets (scene.update); not part of the solve
  check_limits    Configuration.check_limits()
  objective       task QP objective assembly (H, c) over all tasks
  limit:<Name>    each limit's compute_qp_inequalities(), keyed by class name
  ineq_stack      np.vstack/np.hstack of the per-limit (G, h) blocks
  problem_ctor    qpsolvers.Problem(...) construction
  solve_build     daqp repack of the problem into solver form (build_time)
  solve_run       daqp C solve (solve_time)
  solve_overhead  solve_problem() wall minus build and run
  integrate       Configuration.integrate_inplace()

Phases are mutually exclusive and (modulo timer overhead) sum to the step total,
so the breakdown doubles as a wall-clock attribution. The objective phase calls
solve_ik._compute_qp_objective directly, so it always reflects the shipped solver;
only the inequality stacking is re-implemented glue, to time each limit separately.
The heavy calls (task and limit methods, the solver) are the real mink and
qpsolvers code.

Usage:
    uv run python benchmarks/profile_ik.py            # all scenes
    uv run python benchmarks/profile_ik.py aloha      # one scene
    uv run python benchmarks/profile_ik.py --steps 5000 --warmup 1000
"""

import argparse
import statistics
import time
from collections import defaultdict

import numpy as np
import qpsolvers
from scenes import DT, SCENES

from mink.solve_ik import _compute_qp_objective

_ns = time.perf_counter_ns

# Fixed phase ordering; limit:<Name> phases are inserted right after "objective".
PHASE_ORDER = [
    "update",
    "check_limits",
    "objective",
    "ineq_stack",
    "problem_ctor",
    "solve_build",
    "solve_run",
    "solve_overhead",
    "integrate",
]


def timed_step(scene, s, t, dt, tasks, limits, damping, timings) -> None:
    configuration = s["configuration"]

    def rec(name, dt_ns):
        timings[name].append(dt_ns * 1e-3)

    t0 = _ns()
    scene.update(s, t)
    rec("update", _ns() - t0)

    t0 = _ns()
    configuration.check_limits(safety_break=False)
    rec("check_limits", _ns() - t0)

    # Task objective (H, c). Calls the real solve_ik assembly so this phase
    # always reflects the shipped implementation (fused W^T W) and cannot drift.
    t0 = _ns()
    H, c = _compute_qp_objective(configuration, tasks, damping)
    rec("objective", _ns() - t0)

    # Inequalities, timed per limit type. Mirrors _compute_qp_inequalities.
    G_list, h_list = [], []
    for limit in limits:
        t0 = _ns()
        ineq = limit.compute_qp_inequalities(configuration, dt)
        rec(f"limit:{type(limit).__name__}", _ns() - t0)
        if not ineq.inactive:
            G_list.append(ineq.G)
            h_list.append(ineq.h)

    t0 = _ns()
    G = np.vstack(G_list) if G_list else None
    h = np.hstack(h_list) if h_list else None
    rec("ineq_stack", _ns() - t0)

    t0 = _ns()
    problem = qpsolvers.Problem(H, c, G, h, None, None)
    rec("problem_ctor", _ns() - t0)

    t0 = _ns()
    result = qpsolvers.solve_problem(problem, solver="daqp")
    solve_wall_us = (_ns() - t0) * 1e-3
    build_us = (result.build_time or 0.0) * 1e6
    run_us = (result.solve_time or 0.0) * 1e6
    timings["solve_build"].append(build_us)
    timings["solve_run"].append(run_us)
    timings["solve_overhead"].append(max(solve_wall_us - build_us - run_us, 0.0))

    if not result.found:
        raise RuntimeError("QP solver did not find a solution")
    assert result.x is not None
    vel = result.x / dt

    t0 = _ns()
    configuration.integrate_inplace(vel, dt)
    rec("integrate", _ns() - t0)


def med_p95(xs):
    s = sorted(xs)
    return statistics.median(s), s[min(int(0.95 * len(s)), len(s) - 1)]


def order_phases(keys) -> list[str]:
    limit_phases = sorted(k for k in keys if k.startswith("limit:"))
    out: list[str] = []
    for k in PHASE_ORDER:
        if k in keys:
            out.append(k)
            if k == "objective":
                out.extend(limit_phases)
    return out + sorted(k for k in keys if k not in out)


def run(key: str, steps: int, warmup: int) -> None:
    scene = SCENES[key]
    s = scene.setup()
    tasks, limits, damping = s["tasks"], s["limits"], s["damping"]

    timings: dict[str, list] = defaultdict(list)
    sink: dict[str, list] = defaultdict(list)
    t = 0.0
    for i in range(warmup + steps):
        timed_step(
            scene, s, t, DT, tasks, limits, damping, timings if i >= warmup else sink
        )
        t += DT

    phases = order_phases(timings.keys())
    medians = {p: med_p95(timings[p]) for p in phases}
    total = sum(m for m, _ in medians.values())

    print(f"\n{'=' * 64}")
    print(f"  {scene.label}   (nv={scene.nv}, {steps} steps after {warmup} warmup)")
    print(f"{'=' * 64}")
    print(f"  {'phase':<28s} {'median':>9s} {'p95':>9s}   {'%':>6s}")
    print(f"  {'-' * 28} {'-' * 9} {'-' * 9}   {'-' * 6}")
    for p in phases:
        m, p95 = medians[p]
        pct = 100.0 * m / total if total else 0.0
        print(f"  {p:<28s} {m:9.1f} {p95:9.1f}   {pct:5.1f}%")
    print(f"  {'-' * 28} {'-' * 9} {'-' * 9}   {'-' * 6}")
    print(f"  {'sum(median)':<28s} {total:9.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "scenario",
        nargs="?",
        choices=list(SCENES),
        help="Scene to profile (default: all).",
    )
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=500)
    args = ap.parse_args()

    keys = [args.scenario] if args.scenario else list(SCENES)
    print(f"\n  per-component IK profile, dt={DT * 1000:.1f} ms (times in us)")
    for key in keys:
        run(key, args.steps, args.warmup)
    print()


if __name__ == "__main__":
    main()
