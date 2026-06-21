"""Shared utilities for the IK benchmark scripts."""

import statistics
from typing import Sequence


def summarize(times_us: Sequence[float]) -> dict:
    """Summary statistics (in microseconds) for a list of per-step times."""
    s = sorted(times_us)
    n = len(s)

    def pct(p: float) -> float:
        return s[min(int(p * n), n - 1)]

    return {
        "n": n,
        "mean": statistics.mean(s),
        "median": statistics.median(s),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "std": statistics.stdev(s) if n > 1 else 0.0,
        "min": s[0],
        "max": s[-1],
    }
