"""End-to-end IK benchmark: per-step wall time through the public mink API.

Measures what a user pays per control step (target update, mink.solve_ik,
integration) using the real public API, with no internal timers. Use it for
regression tracking and A/B comparison between branches.

Usage:
    uv run python benchmarks/bench_ik.py                          # all scenes
    uv run python benchmarks/bench_ik.py aloha --save before.json # one scene, save
    uv run python benchmarks/bench_ik.py --compare a.json b.json  # B/A speedup
"""

import argparse
import json
import time
from pathlib import Path

from common import summarize
from scenes import DT, SCENES

import mink

_ns = time.perf_counter_ns


def run_scene(key: str, steps: int, warmup: int, solver: str) -> dict:
    scene = SCENES[key]
    s = scene.setup()
    configuration = s["configuration"]
    tasks, limits, damping = s["tasks"], s["limits"], s["damping"]

    times_us: list[float] = []
    t = 0.0
    for i in range(warmup + steps):
        t0 = _ns()
        scene.update(s, t)
        vel = mink.solve_ik(
            configuration, tasks, DT, solver, damping=damping, limits=limits
        )
        configuration.integrate_inplace(vel, DT)
        if i >= warmup:
            times_us.append((_ns() - t0) * 1e-3)
        t += DT
    return summarize(times_us)


_FIELDS = ("mean", "median", "p95", "p99", "std", "min", "max")


def print_stats(label: str, stats: dict) -> None:
    print(f"\n  {label}  (n={stats['n']})")
    for f in _FIELDS:
        print(f"    {f:>7s}: {stats[f]:9.1f} us")


def print_comparison(a: dict, b: dict, la: str, lb: str) -> None:
    print(f"\n  {'metric':<8s} {la:>12s} {lb:>12s} {'speedup':>9s}")
    print(f"  {'-' * 8} {'-' * 12} {'-' * 12} {'-' * 9}")
    for m in ("mean", "median", "p95", "p99"):
        x, y = a[m], b[m]
        speedup = x / y if y else float("inf")
        print(f"  {m:<8s} {x:12.1f} {y:12.1f} {speedup:8.2f}x")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "scenario",
        nargs="?",
        choices=list(SCENES),
        help="Scene to benchmark (default: all).",
    )
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--solver", default="daqp")
    ap.add_argument("--save", type=str, help="Write results to a JSON file.")
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("A", "B"),
        help="Print B-relative-to-A speedup for two saved JSON files.",
    )
    args = ap.parse_args()

    if args.compare:
        a = json.loads(Path(args.compare[0]).read_text())
        b = json.loads(Path(args.compare[1]).read_text())
        la, lb = Path(args.compare[0]).stem, Path(args.compare[1]).stem
        print("\n  IK benchmark A/B comparison")
        for k in sorted(set(a) & set(b)):
            print(f"\n== {k} ==")
            print_comparison(a[k], b[k], la, lb)
        print()
        return

    keys = [args.scenario] if args.scenario else list(SCENES)
    print(
        f"\n  end-to-end IK, solver={args.solver}, dt={DT * 1000:.1f} ms, "
        f"{args.steps} steps after {args.warmup} warmup"
    )

    results = {}
    for key in keys:
        stats = run_scene(key, args.steps, args.warmup, args.solver)
        results[key] = stats
        print_stats(SCENES[key].label, stats)
    print()

    if args.save:
        Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"  saved -> {args.save}\n")


if __name__ == "__main__":
    main()
