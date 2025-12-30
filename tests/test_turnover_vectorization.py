from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from trend_analysis.multi_period.engine import _compute_turnover_state
from trend_analysis.rebalancing.strategies import TURNOVER_EPSILON, TurnoverCapStrategy


def python_turnover_state(
    prev_idx: np.ndarray | None,
    prev_vals: np.ndarray | None,
    new_series: pd.Series,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Reference implementation mirroring the former Python loop."""

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


@pytest.mark.parametrize("cols", [6, 12])
def test_vectorised_turnover_matches_python(cols: int) -> None:
    rng = np.random.default_rng(12345)
    assets = np.array([f"F{i:03d}" for i in range(cols)], dtype=object)

    prev_idx_py: np.ndarray | None = None
    prev_vals_py: np.ndarray | None = None
    prev_idx_vec: np.ndarray | None = None
    prev_vals_vec: np.ndarray | None = None

    for _ in range(50):
        size = int(rng.integers(1, cols + 1))
        chosen = rng.choice(assets, size=size, replace=False)
        values = rng.random(size)
        weights = pd.Series(values / values.sum(), index=chosen)

        expected, prev_idx_py, prev_vals_py = python_turnover_state(
            prev_idx_py, prev_vals_py, weights
        )
        got, prev_idx_vec, prev_vals_vec = _compute_turnover_state(
            prev_idx_vec, prev_vals_vec, weights
        )
        assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12)
        assert np.array_equal(prev_idx_vec, weights.index.to_numpy())
        np.testing.assert_allclose(prev_vals_vec, weights.to_numpy(), rtol=0, atol=0)


def python_turnover_cap(
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
        priorities = strategy._calculate_priorities(current_aligned, target_aligned, trades, scores)
        trade_items = [
            (asset, trade, priority)
            for asset, trade, priority in zip(trades.index, trades.values, priorities.values)
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


@pytest.mark.parametrize("priority", ["largest_gap", "best_score_delta"])
def test_turnover_cap_vectorisation_matches_python(priority: str) -> None:
    rng = np.random.default_rng(54321)
    cols = 10
    assets = np.array([f"A{i:03d}" for i in range(cols)], dtype=object)
    params = {"max_turnover": 0.35, "cost_bps": 15, "priority": priority}

    for _ in range(25):
        cur_size = int(rng.integers(1, cols + 1))
        tgt_size = int(rng.integers(1, cols + 1))
        cur_idx = rng.choice(assets, size=cur_size, replace=False)
        tgt_idx = rng.choice(assets, size=tgt_size, replace=False)

        cur_vals = rng.random(cur_size)
        tgt_vals = rng.random(tgt_size)
        current = pd.Series(cur_vals / cur_vals.sum(), index=cur_idx)
        target = pd.Series(tgt_vals / tgt_vals.sum(), index=tgt_idx)

        scores = None
        if priority == "best_score_delta":
            scores = pd.Series(rng.random(cols), index=assets)

        strategy_py = TurnoverCapStrategy(params)
        strategy_vec = TurnoverCapStrategy(params)

        expected_weights, expected_cost = python_turnover_cap(strategy_py, current, target, scores)
        actual_weights, actual_cost = strategy_vec.apply(current, target, scores=scores)

        pd.testing.assert_series_equal(
            actual_weights.sort_index(),
            expected_weights.sort_index(),
            check_names=False,
            rtol=1e-12,
            atol=1e-12,
        )
        assert math.isclose(actual_cost, expected_cost, rel_tol=1e-12, abs_tol=1e-12)
