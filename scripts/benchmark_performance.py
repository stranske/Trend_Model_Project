#!/usr/bin/env python3
"""Benchmark covariance metric computation with and without cache.

Outputs JSON summary with timing statistics. This is a micro‑benchmark
focused on the synthetic '__COV_VAR__' metric used to exercise the
CovCache path.

Usage:
  python scripts/benchmark_performance.py --rows 2000 --cols 50 --runs 5
"""
from __future__ import annotations

import argparse
import json
import time
from statistics import mean

import numpy as np
import pandas as pd

from trend_analysis.perf.cache import CovCache
from trend_analysis.core.rank_selection import compute_metric_series_with_cache
from trend_analysis.metrics import RiskStatsConfig


def _make_df(rows: int, cols: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(scale=0.01, size=(rows, cols))
    cols_ = [f"F{i:03d}" for i in range(cols)]
    return pd.DataFrame(data, columns=cols_)


def run_benchmark(rows: int, cols: int, runs: int) -> dict:
    df = _make_df(rows, cols)
    stats_cfg = RiskStatsConfig(periods_per_year=252, risk_free=0.0)
    timings_no_cache = []
    timings_cache = []
    cache = CovCache()

    for _ in range(runs):
        t0 = time.perf_counter()
        compute_metric_series_with_cache(
            df, "__COV_VAR__", stats_cfg, enable_cache=False
        )
        timings_no_cache.append(time.perf_counter() - t0)

    for _ in range(runs):
        t0 = time.perf_counter()
        compute_metric_series_with_cache(
            df, "__COV_VAR__", stats_cfg, cov_cache=cache, enable_cache=True
        )
        timings_cache.append(time.perf_counter() - t0)

    return {
        "rows": rows,
        "cols": cols,
        "runs": runs,
        "no_cache_mean_s": mean(timings_no_cache),
        "cache_mean_s": mean(timings_cache),
        "speedup_x": (
            (mean(timings_no_cache) / mean(timings_cache))
            if mean(timings_cache) > 0
            else None
        ),
        "detail": {
            "no_cache": timings_no_cache,
            "cache": timings_cache,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=1500)
    p.add_argument("--cols", type=int, default=40)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--output", type=str, default="benchmark_perf.json")
    args = p.parse_args()

    result = run_benchmark(args.rows, args.cols, args.runs)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
