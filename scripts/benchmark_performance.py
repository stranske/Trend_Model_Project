#!/usr/bin/env python3
"""Benchmark core performance hotspots and vectorisation speedups.

Outputs JSON summary with timing statistics covering:

* Covariance metric computation with and without ``CovCache``.
* Turnover computation inside the multi-period engine (Python loop vs
  vectorised path).
* Turnover cap rebalancing (loop-based allocation vs vectorised priority
  execution).

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

from trend_analysis.core.rank_selection import (
    RiskStatsConfig,
    compute_metric_series_with_cache,
)
from trend_analysis.multi_period.engine import _compute_turnover_state
from trend_analysis.perf.cache import CovCache
from trend_analysis.rebalancing.strategies import TURNOVER_EPSILON, TurnoverCapStrategy


def _make_df(rows: int, cols: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(scale=0.01, size=(rows, cols))
    cols_ = [f"F{i:03d}" for i in range(cols)]
    return pd.DataFrame(data, columns=cols_)


def _python_turnover_state(
    prev_idx: np.ndarray | None,
    prev_vals: np.ndarray | None,
    new_series: pd.Series,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Original Python loop implementation used for benchmarking."""

    nidx = new_series.index.to_numpy()
    nvals = new_series.to_numpy(dtype=float, copy=True)
    if prev_idx is None or prev_vals is None:
        return float(np.abs(nvals).sum()), nidx, nvals

    pmap = {k: i for i, k in enumerate(prev_idx.tolist())}
    union_list: list[str] = []
    seen: set[str] = set()
    for key in nidx.tolist():
        union_list.append(key)
        seen.add(key)
    for key in prev_idx.tolist():
        if key not in seen:
            union_list.append(key)
            seen.add(key)
    union_arr = np.array(union_list, dtype=object)
    new_aligned = np.zeros(len(union_arr), dtype=float)
    prev_aligned = np.zeros(len(union_arr), dtype=float)
    nmap = {k: i for i, k in enumerate(nidx.tolist())}
    for i, key in enumerate(union_arr.tolist()):
        if key in nmap:
            new_aligned[i] = nvals[nmap[key]]
        if key in pmap:
            prev_aligned[i] = prev_vals[pmap[key]]
    turnover = float(np.abs(new_aligned - prev_aligned).sum())
    return turnover, nidx, nvals


def _generate_weight_series(
    rng: np.random.Generator, rows: int, cols: int
) -> list[pd.Series]:
    assets = np.array([f"F{i:03d}" for i in range(cols)], dtype=object)
    series: list[pd.Series] = []
    for _ in range(rows):
        size = int(rng.integers(1, cols + 1))
        chosen = rng.choice(assets, size=size, replace=False)
        values = rng.random(size)
        weights = values / values.sum()
        series.append(pd.Series(weights, index=chosen))
    return series


def _benchmark_turnover_alignment(rows: int, cols: int, runs: int) -> dict:
    rng = np.random.default_rng(20240518)
    python_times: list[float] = []
    vector_times: list[float] = []

    for _ in range(runs):
        weight_history = _generate_weight_series(rng, rows, cols)

        t0 = time.perf_counter()
        prev_idx_py: np.ndarray | None = None
        prev_vals_py: np.ndarray | None = None
        total_py = 0.0
        for series in weight_history:
            turnover, prev_idx_py, prev_vals_py = _python_turnover_state(
                prev_idx_py, prev_vals_py, series
            )
            total_py += turnover
        python_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        prev_idx_vec: np.ndarray | None = None
        prev_vals_vec: np.ndarray | None = None
        total_vec = 0.0
        for series in weight_history:
            turnover, prev_idx_vec, prev_vals_vec = _compute_turnover_state(
                prev_idx_vec, prev_vals_vec, series
            )
            total_vec += turnover
        vector_times.append(time.perf_counter() - t1)

        if not np.isclose(total_py, total_vec, rtol=1e-12, atol=1e-12):
            raise RuntimeError(
                "Turnover totals diverged between Python and vectorised implementations"
            )

    mean_python = mean(python_times)
    mean_vector = mean(vector_times)
    return {
        "python_mean_s": mean_python,
        "vectorized_mean_s": mean_vector,
        "speedup_x": (mean_python / mean_vector) if mean_vector > 0 else None,
        "detail": {"python": python_times, "vectorized": vector_times},
    }


def _python_turnover_cap(
    strategy: TurnoverCapStrategy,
    current: pd.Series,
    target: pd.Series,
    scores: pd.Series | None,
) -> tuple[pd.Series, float]:
    all_assets = current.index.union(target.index)
    current_aligned = current.reindex(all_assets, fill_value=0.0)
    target_aligned = target.reindex(all_assets, fill_value=0.0)
    trades = target_aligned - current_aligned
    total_desired = trades.abs().sum()
    if total_desired <= strategy.max_turnover:
        actual_turnover = float(total_desired)
        new_weights = target_aligned.copy()
    else:
        priorities = strategy._calculate_priorities(
            current_aligned, target_aligned, trades, scores
        )
        trade_items = [
            (asset, trade, priority)
            for asset, trade, priority in zip(
                trades.index, trades.values, priorities.values
            )
        ]
        trade_items.sort(key=lambda x: x[2], reverse=True)

        remaining = strategy.max_turnover
        executed = pd.Series(0.0, index=trades.index)
        for asset, desired_trade, _ in trade_items:
            if remaining <= TURNOVER_EPSILON:
                break
            trade_size = abs(desired_trade)
            if trade_size <= remaining + TURNOVER_EPSILON:
                executed[asset] = desired_trade
                remaining -= trade_size
            elif desired_trade != 0:
                scale_factor = remaining / trade_size
                executed[asset] = desired_trade * scale_factor
                remaining = 0.0
        new_weights = (current_aligned + executed).clip(lower=0.0)
        actual_turnover = float(executed.abs().sum())

    cost = strategy._calculate_cost(actual_turnover)
    return new_weights, cost


def _generate_rebalance_cases(
    rng: np.random.Generator, cols: int, scenarios: int
) -> list[tuple[pd.Series, pd.Series, pd.Series | None]]:
    assets = np.array([f"F{i:03d}" for i in range(cols)], dtype=object)
    cases: list[tuple[pd.Series, pd.Series, pd.Series | None]] = []
    for _ in range(scenarios):
        cur_size = int(rng.integers(1, cols + 1))
        tgt_size = int(rng.integers(1, cols + 1))
        cur_idx = rng.choice(assets, size=cur_size, replace=False)
        tgt_idx = rng.choice(assets, size=tgt_size, replace=False)
        cur_vals = rng.random(cur_size)
        tgt_vals = rng.random(tgt_size)
        current = pd.Series(cur_vals / cur_vals.sum(), index=cur_idx)
        target = pd.Series(tgt_vals / tgt_vals.sum(), index=tgt_idx)
        scores = pd.Series(rng.random(len(assets)), index=assets)
        cases.append((current, target, scores))
    return cases


def _benchmark_turnover_cap(cols: int, scenarios: int, runs: int) -> dict:
    rng = np.random.default_rng(20240519)
    results: dict[str, dict[str, object]] = {}

    for priority in ("largest_gap", "best_score_delta"):
        python_times: list[float] = []
        vector_times: list[float] = []

        for _ in range(runs):
            cases = _generate_rebalance_cases(rng, cols, scenarios)
            params = {"max_turnover": 0.25, "cost_bps": 10, "priority": priority}
            strat_python = TurnoverCapStrategy(params)
            strat_vector = TurnoverCapStrategy(params)

            t0 = time.perf_counter()
            for current, target, scores in cases:
                score_input = scores if priority == "best_score_delta" else None
                _python_turnover_cap(strat_python, current, target, score_input)
            python_times.append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            for current, target, scores in cases:
                score_input = scores if priority == "best_score_delta" else None
                strat_vector.apply(current, target, scores=score_input)
            vector_times.append(time.perf_counter() - t1)

        mean_python = mean(python_times)
        mean_vector = mean(vector_times)
        results[priority] = {
            "python_mean_s": mean_python,
            "vectorized_mean_s": mean_vector,
            "speedup_x": (mean_python / mean_vector) if mean_vector > 0 else None,
            "detail": {"python": python_times, "vectorized": vector_times},
        }

    return results


def run_benchmark(rows: int, cols: int, runs: int) -> dict:
    df = _make_df(rows, cols)
    stats_cfg = RiskStatsConfig(periods_per_year=252, risk_free=0.0)
    timings_no_cache: list[float] = []
    timings_cache: list[float] = []
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

    turnover_alignment = _benchmark_turnover_alignment(rows, cols, runs)
    turnover_cap = _benchmark_turnover_cap(cols, rows, runs)

    mean_no_cache = mean(timings_no_cache)
    mean_cache = mean(timings_cache)

    return {
        "rows": rows,
        "cols": cols,
        "runs": runs,
        "no_cache_mean_s": mean_no_cache,
        "cache_mean_s": mean_cache,
        "speedup_x": (mean_no_cache / mean_cache) if mean_cache > 0 else None,
        "detail": {
            "no_cache": timings_no_cache,
            "cache": timings_cache,
        },
        "turnover_vectorization": turnover_alignment,
        "turnover_cap_vectorization": turnover_cap,
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
    from trend_analysis.script_logging import setup_script_logging

    setup_script_logging(module_file=__file__)
    main()
