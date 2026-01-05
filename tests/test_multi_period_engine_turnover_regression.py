"""Regression tests for turnover and weight bound handling.

These tests enforce invariants around weight normalisation, bound clamping,
and turnover bookkeeping for the multi-period engine helpers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from trend_analysis.config import Config
from trend_analysis.multi_period import engine as mp_engine
from trend_analysis.multi_period import run as run_mp
from trend_analysis.weighting import BaseWeighting


def test_apply_weight_bounds_clamps_and_normalises() -> None:
    """Bounds application must prevent negatives and preserve unit sum."""

    weights = pd.Series({"A": 0.8, "B": 0.2, "C": 0.1})

    bounded = mp_engine._apply_weight_bounds(weights, min_w_bound=0.05, max_w_bound=0.6)

    assert bounded.between(0.05 - 1e-12, 0.6 + 1e-12).all()
    assert bounded.sum() == pytest.approx(1.0, rel=0, abs=1e-12)
    assert bounded.idxmax() == "A"

    # Negative inputs and sub-minimum weights are lifted then redistributed.
    weights_with_negatives = pd.Series({"A": -0.2, "B": 0.93, "C": 0.05})

    bounded_negative = mp_engine._apply_weight_bounds(
        weights_with_negatives, min_w_bound=0.1, max_w_bound=0.7
    )

    assert (bounded_negative >= 0).all()
    assert bounded_negative.between(0.1 - 1e-12, 0.7 + 1e-12).all()
    assert bounded_negative.sum() == pytest.approx(1.0, rel=0, abs=1e-12)


def test_turnover_penalty_respects_bounds_and_normalisation() -> None:
    """Penalised turnover must stay within bounds and remain normalised."""

    last = pd.Series({"A": 0.9, "B": 0.05, "C": 0.05}, dtype=float)
    target = pd.Series({"A": 0.2, "B": 0.2, "C": 0.6}, dtype=float)

    adjusted = mp_engine._apply_turnover_penalty(
        target,
        last,
        lambda_tc=0.4,
        min_w_bound=0.1,
        max_w_bound=0.6,
    )

    assert adjusted.between(0.1 - 1e-12, 0.6 + 1e-12).all()
    assert adjusted.sum() == pytest.approx(1.0, rel=0, abs=1e-12)
    assert (adjusted >= 0).all()

    # Shrinkage should move allocations toward the previous weights while still
    # respecting the hard bounds after redistribution.
    shrunk = last + (target - last) * 0.6
    assert adjusted.idxmax() == shrunk.idxmax()


class _FixedWeighting(BaseWeighting):
    def __init__(self, sequences: list[pd.Series]):
        self._sequences = sequences
        self._call = 0

    def weight(
        self, selected: pd.DataFrame, date: pd.Timestamp
    ) -> pd.DataFrame:  # pragma: no cover - simple
        del selected, date
        series = self._sequences[self._call]
        self._call += 1
        return series.to_frame("weight")


def test_fast_turnover_matches_recomputed_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fast turnover path should agree with recomputed turnover history."""

    frames = {
        "2020-01-31": pd.DataFrame({"score": [1.0, 0.5]}, index=["A", "B"]),
        "2020-02-29": pd.DataFrame({"score": [0.7, 0.3]}, index=["B", "C"]),
        "2020-03-31": pd.DataFrame({"score": [0.9, 0.1]}, index=["C", "D"]),
    }

    returned: list[pd.Series] = []
    call_idx = {"i": 0}

    def fake_apply(
        strategies,  # pragma: no cover - signature compatibility
        params,
        current: pd.Series,
        target: pd.Series,
        *,
        scores: pd.Series | None = None,
    ) -> tuple[pd.Series, float]:
        del strategies, params, current, target, scores
        series = returned[call_idx["i"]]
        call_idx["i"] += 1
        return series, 0.0

    returned.extend(
        [
            pd.Series({"A": 0.6, "B": 0.4}, dtype=float),
            pd.Series({"B": 0.2, "C": 0.8}, dtype=float),
            pd.Series({"C": 0.5, "D": 0.5}, dtype=float),
        ]
    )

    class DummySelector:
        def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            return score_frame, score_frame

    weighting = _FixedWeighting(returned)

    monkeypatch.setattr(mp_engine, "apply_rebalancing_strategies", fake_apply)

    portfolio = mp_engine.run_schedule(
        frames,
        DummySelector(),
        weighting,
        rebalance_strategies=["dummy"],
        rebalance_params={"dummy": {}},
    )

    assert list(portfolio.history) == ["2020-01-31", "2020-02-29", "2020-03-31"]

    recomputed: dict[str, float] = {}
    prev: pd.Series | None = None
    for date_key in sorted(portfolio.history):
        weights = portfolio.history[date_key].astype(float)
        if prev is None:
            recomputed[date_key] = float(np.abs(weights).sum())
        else:
            union = prev.index.union(weights.index)
            prev_aligned = prev.reindex(union, fill_value=0.0)
            new_aligned = weights.reindex(union, fill_value=0.0)
            recomputed[date_key] = float(np.abs(new_aligned - prev_aligned).sum())
        prev = weights

    for date_key, expected in recomputed.items():
        assert portfolio.turnover[date_key] == pytest.approx(expected)


def _turnover_series(results: list[dict[str, object]]) -> list[float]:
    values: list[float] = []
    for res in results:
        risk_diag = res.get("risk_diagnostics", {})
        if isinstance(risk_diag, dict):
            turnover = risk_diag.get("turnover")
        else:
            turnover = None
        if isinstance(turnover, pd.Series) and not turnover.empty:
            values.append(float(turnover.iloc[-1]))
        elif isinstance(turnover, (int, float)):
            values.append(float(turnover))
    return values


def _turnover_config(max_turnover: float) -> Config:
    cfg_data = yaml.safe_load(Path("config/defaults.yml").read_text())
    cfg_data["multi_period"] = {
        "frequency": "M",
        "in_sample_len": 2,
        "out_sample_len": 1,
        "start": "2020-01",
        "end": "2020-04",
    }
    data_cfg = cfg_data.setdefault("data", {})
    data_cfg.setdefault("date_column", "Date")
    data_cfg.setdefault("frequency", "M")
    data_cfg["allow_risk_free_fallback"] = True
    portfolio_cfg = cfg_data.setdefault("portfolio", {})
    portfolio_cfg["policy"] = ""
    portfolio_cfg["selection_mode"] = "rank"
    portfolio_cfg["rank"] = {"inclusion_approach": "top_n", "n": 1}
    portfolio_cfg["weighting_scheme"] = "equal"
    portfolio_cfg["constraints"] = {"max_weight": 1.0, "long_only": True}
    portfolio_cfg["max_turnover"] = max_turnover
    return Config(**cfg_data)


def test_multi_period_turnover_cap_reduces_rebalance_size() -> None:
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "A": [0.05, 0.05, -0.02, 0.01],
            "B": [0.01, -0.02, 0.06, -0.01],
        }
    )

    high_cap_cfg = _turnover_config(1.0)
    low_cap_cfg = _turnover_config(0.3)

    high_results = run_mp(high_cap_cfg, df)
    low_results = run_mp(low_cap_cfg, df)

    high_turnover = _turnover_series(high_results)
    low_turnover = _turnover_series(low_results)

    assert len(high_turnover) == len(low_turnover) >= 2
    assert low_turnover[1] <= 0.3 + 1e-12
    assert low_turnover[1] < high_turnover[1]
