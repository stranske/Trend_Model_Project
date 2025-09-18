#!/usr/bin/env python3
"""Compare benchmark output against stored baseline and enforce regression budget.

The script inspects the JSON emitted by ``benchmark_performance.py`` and
validates that runtime metrics have not regressed beyond an allowed
percentage.  It focuses on the high-level summary values that represent the
core hotspots we care about:

* Covariance calculations (cache vs. no-cache timings and speedup).
* Turnover alignment vectorisation timings.
* Turnover-cap rebalancing timings for the supported priority modes.

Each metric is categorised as either "runtime" (smaller-is-better) or
"speedup" (larger-is-better).  The regression budget defaults to 15%, but can
be overridden via the ``PERF_REGRESSION_PCT`` environment variable or the
``--threshold`` CLI option.  Threshold values greater than 1 are interpreted
as percentages (e.g. 15 == 15%), while values between 0 and 1 are treated as
already expressed in fractional form (e.g. 0.15 == 15%).

The script exits with status 1 when any metric breaches the budget, printing a
human-readable summary to stderr while still emitting a JSON summary to stdout
for downstream tooling.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable


MetricType = str


@dataclass(frozen=True)
class Metric:
    """Represents a comparable metric extracted from a benchmark run."""

    key: str
    value: float
    baseline: float
    metric_type: MetricType  # "runtime" | "speedup"

    def regression_pct(self) -> float:
        """Return the regression magnitude as a percentage."""

        if math.isclose(self.baseline, 0.0):
            # Degenerate case where baseline is zero; any deviation counts as
            # either a regression (runtime getting slower) or improvement.
            if self.metric_type == "runtime":
                return 100.0 if self.value > 0 else 0.0
            return 100.0 if self.value < 0 else 0.0

        change_ratio = (self.value - self.baseline) / self.baseline

        if self.metric_type == "runtime":
            return max(change_ratio * 100.0, 0.0)

        # For speedups we treat decreases as regressions.
        result = max(-change_ratio, 0.0) * 100.0
        return 0.0 if abs(result) < 1e-12 else result

    def is_regression(self, threshold_pct: float) -> bool:
        return self.regression_pct() > threshold_pct


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalise_threshold(raw_threshold: float) -> float:
    if raw_threshold < 0:
        raise ValueError("Threshold must be non-negative")
    if raw_threshold <= 1:
        return raw_threshold * 100.0
    return raw_threshold


def _iter_metrics(data: dict, baseline: dict) -> Iterable[Metric]:
    yield Metric(
        key="covariance.cache_mean_s",
        value=float(data["cache_mean_s"]),
        baseline=float(baseline["cache_mean_s"]),
        metric_type="runtime",
    )
    yield Metric(
        key="covariance.speedup_x",
        value=float(data["speedup_x"]),
        baseline=float(baseline["speedup_x"]),
        metric_type="speedup",
    )

    turn_data = data.get("turnover_vectorization", {})
    turn_baseline = baseline.get("turnover_vectorization", {})
    yield Metric(
        key="turnover.vectorized_mean_s",
        value=float(turn_data["vectorized_mean_s"]),
        baseline=float(turn_baseline["vectorized_mean_s"]),
        metric_type="runtime",
    )
    yield Metric(
        key="turnover.speedup_x",
        value=float(turn_data["speedup_x"]),
        baseline=float(turn_baseline["speedup_x"]),
        metric_type="speedup",
    )

    cap_data = data.get("turnover_cap_vectorization", {})
    cap_baseline = baseline.get("turnover_cap_vectorization", {})
    for priority in sorted(cap_data):
        key_prefix = f"turnover_cap.{priority}"
        current = cap_data[priority]
        base = cap_baseline.get(priority, {})
        yield Metric(
            key=f"{key_prefix}.vectorized_mean_s",
            value=float(current["vectorized_mean_s"]),
            baseline=float(base["vectorized_mean_s"]),
            metric_type="runtime",
        )
        yield Metric(
            key=f"{key_prefix}.speedup_x",
            value=float(current["speedup_x"]),
            baseline=float(base["speedup_x"]),
            metric_type="speedup",
        )


def _threshold_from_env(default_pct: float) -> float:
    env_val = os.getenv("PERF_REGRESSION_PCT")
    if not env_val:
        return default_pct
    try:
        parsed = float(env_val)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(
            "PERF_REGRESSION_PCT must be a numeric value"
        ) from exc
    return parsed


def compare_metrics(
    benchmark_path: str, baseline_path: str, threshold_pct: float
) -> dict:
    data = _load_json(benchmark_path)
    baseline = _load_json(baseline_path)
    metrics = list(_iter_metrics(data, baseline))

    threshold = _normalise_threshold(threshold_pct)
    results: list[dict[str, object]] = []
    regressions: list[Metric] = []

    for metric in metrics:
        regression_pct = metric.regression_pct()
        is_reg = metric.is_regression(threshold)
        if is_reg:
            regressions.append(metric)
        results.append(
            {
                "key": metric.key,
                "value": metric.value,
                "baseline": metric.baseline,
                "type": metric.metric_type,
                "regression_pct": regression_pct,
                "threshold_pct": threshold,
                "status": "fail" if is_reg else "ok",
            }
        )

    return {
        "benchmark_path": benchmark_path,
        "baseline_path": baseline_path,
        "threshold_pct": threshold,
        "metrics": results,
        "regressions": [m.key for m in regressions],
    }


def _print_summary(result: dict) -> None:
    header = (
        f"Performance comparison (threshold {result['threshold_pct']:.2f}%):"
    )
    print(header, file=sys.stderr)
    for metric in result["metrics"]:
        status = metric["status"]
        regression_pct = metric["regression_pct"]
        print(
            f"  [{status.upper():<4}] {metric['key']}: "
            f"value={metric['value']:.6f}, baseline={metric['baseline']:.6f}, "
            f"regression={regression_pct:.2f}%",
            file=sys.stderr,
        )

    if result["regressions"]:
        print("Detected regressions:", file=sys.stderr)
        for key in result["regressions"]:
            print(f"  - {key}", file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare benchmark output against the stored baseline",
    )
    parser.add_argument(
        "--benchmark",
        default="benchmark_perf.json",
        help="Path to the newly generated benchmark JSON",
    )
    parser.add_argument(
        "--baseline",
        default="perf_baseline.json",
        help="Path to the baseline JSON file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Regression threshold in percent. Overrides PERF_REGRESSION_PCT "
            "when provided."
        ),
    )
    return parser.parse_args()


def main() -> int:  # pragma: no cover - thin CLI wrapper
    args = _parse_args()
    env_threshold = _threshold_from_env(15.0)
    threshold = args.threshold if args.threshold is not None else env_threshold

    try:
        result = compare_metrics(args.benchmark, args.baseline, threshold)
    except FileNotFoundError as exc:
        print(f"Missing file: {exc.filename}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"Error while comparing benchmarks: {exc}", file=sys.stderr)
        return 3

    _print_summary(result)
    print(json.dumps(result, indent=2))

    return 1 if result["regressions"] else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

