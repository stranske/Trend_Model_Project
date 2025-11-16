#!/usr/bin/env python3
"""Compare current benchmark JSON against baseline with regression budget.

Usage:
  python scripts/compare_perf.py --current results.json --baseline perf/perf_baseline.json \
      --threshold-pct 15

Non-strict: regression must be strictly greater than threshold to fail.
Monitored raw time metrics (nested keys flattened with '.'):
  - no_cache_mean_s
  - cache_mean_s
  - turnover_vectorization.python_mean_s
  - turnover_vectorization.vectorized_mean_s
  - turnover_cap_vectorization.largest_gap.python_mean_s
  - turnover_cap_vectorization.largest_gap.vectorized_mean_s
  - turnover_cap_vectorization.best_score_delta.python_mean_s
  - turnover_cap_vectorization.best_score_delta.vectorized_mean_s

Exit codes:
  0 - success within budget
  1 - regression exceeded threshold or structural mismatch
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

MONITORED = [
    "no_cache_mean_s",
    "cache_mean_s",
    "turnover_vectorization.python_mean_s",
    "turnover_vectorization.vectorized_mean_s",
    "turnover_cap_vectorization.largest_gap.python_mean_s",
    "turnover_cap_vectorization.largest_gap.vectorized_mean_s",
    "turnover_cap_vectorization.best_score_delta.python_mean_s",
    "turnover_cap_vectorization.best_score_delta.vectorized_mean_s",
]


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def compare(
    current: Dict[str, Any], baseline: Dict[str, Any], threshold_pct: float
) -> int:
    flat_cur = _flatten(current)
    flat_base = _flatten(baseline)

    missing: List[str] = []
    regressions: List[tuple[str, float, float, float]] = []

    for metric in MONITORED:
        if metric not in flat_base:
            print(f"WARNING: baseline missing metric '{metric}', skipping gating")
            continue
        if metric not in flat_cur:
            missing.append(metric)
            continue
        b = flat_base[metric]
        c = flat_cur[metric]
        if b <= 0 or math.isnan(b):
            print(f"INFO: baseline {metric} non-positive or NaN, skip")
            continue
        pct = (c - b) / b * 100.0
        status = "OK"
        if pct > threshold_pct:  # strictly greater triggers fail
            status = "REGRESSION"
            regressions.append((metric, b, c, pct))
        print(
            f"{metric:70s} baseline={b:.6g} current={c:.6g} delta={pct:+.2f}% {status}"
        )

    if missing:
        print("ERROR: missing metrics in current run:", ", ".join(missing))
        return 1
    if regressions:
        print("\nFAILED: Performance regressions exceeding threshold:")
        for m, b, c, pct in regressions:
            print(
                f"  {m}: baseline={b:.6g} current={c:.6g} delta={pct:+.2f}% > threshold"
            )
        return 1
    print("\nSUCCESS: All monitored metrics within regression budget.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--current", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument(
        "--threshold-pct",
        type=float,
        default=float(
            Path.cwd().joinpath(".env").read_text().strip()
            if False
            else 15.0  # placeholder
        ),
    )
    args = p.parse_args()

    with open(args.current, "r", encoding="utf-8") as f:
        current = json.load(f)
    with open(args.baseline, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    code = compare(current, baseline, args.threshold_pct)
    sys.exit(code)


if __name__ == "__main__":  # pragma: no cover
    from trend_analysis.script_logging import setup_script_logging

    setup_script_logging(module_file=__file__)
    main()
