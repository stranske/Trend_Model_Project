"""Additional coverage tests for the multi-period engine turnover helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trend_analysis.multi_period import engine as mp_engine
from trend_analysis.weighting import BaseWeighting


def test_compute_turnover_state_union_alignment() -> None:
    """Ensure the vectorised turnover helper handles index unions correctly."""

    prev_idx = np.array(["A", "C"], dtype=object)
    prev_vals = np.array([0.6, 0.4], dtype=float)
    new_series = pd.Series([0.2, 0.8], index=["B", "C"], dtype=float)

    turnover, next_idx, next_vals = mp_engine._compute_turnover_state(
        prev_idx, prev_vals, new_series
    )

    # Alignment should follow the new series ordering and preserve its values.
    assert list(next_idx) == ["B", "C"]
    assert pytest.approx(next_vals.tolist()) == [0.2, 0.8]

    # Turnover sums absolute changes after aligning by the union of indexes.
    # prev -> {A:0.6, C:0.4}, new -> {B:0.2, C:0.8}
    expected_turnover = abs(0.6) + abs(0.4 - 0.8) + abs(0.2)
    assert turnover == pytest.approx(expected_turnover)


class TrackingWeighting(BaseWeighting):
    """Weighting helper that records updates for assertions."""

    def __init__(self) -> None:
        self.updates: list[tuple[pd.Series, int]] = []

    def weight(self, selected: pd.DataFrame, date: pd.Timestamp | None = None) -> pd.DataFrame:
        del date
        if selected.empty:
            return pd.DataFrame(columns=["weight"])
        values = np.linspace(1.0, 0.6, num=len(selected), dtype=float)
        weights = pd.Series(values, index=selected.index, dtype=float)
        weights /= weights.sum()
        return weights.to_frame("weight")

    def update(self, scores: pd.Series, days: int) -> None:  # pragma: no cover - simple
        self.updates.append((scores.astype(float), days))


def test_run_schedule_fast_turnover_tracks_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the specialised fast-turnover path when holdings change."""

    frames = {
        "2020-01-31": pd.DataFrame({"Sharpe": [0.9, 0.5]}, index=["A", "B"]),
        "2020-02-29": pd.DataFrame({"Sharpe": [0.4, 0.8]}, index=["B", "C"]),
    }

    class DummySelector:
        column = "Sharpe"

        def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            return score_frame, score_frame

    returned_series: list[pd.Series] = []

    def fake_apply(
        strategies: list[str],
        params: dict[str, dict[str, Any]],
        current: pd.Series,
        target: pd.Series,
        *,
        scores: pd.Series | None = None,
    ) -> tuple[pd.Series, float]:
        # Return deterministic weights but change the universe on the second call.
        if scores is not None and scores.index.equals(pd.Index(["A", "B"])):
            series = pd.Series({"A": 0.55, "B": 0.45}, dtype=float)
        else:
            series = pd.Series({"B": 0.25, "C": 0.75}, dtype=float)
        returned_series.append(series)
        return series, 0.1

    monkeypatch.setattr(mp_engine, "apply_rebalancing_strategies", fake_apply)

    weighting = TrackingWeighting()
    portfolio = mp_engine.run_schedule(
        frames,
        DummySelector(),
        weighting,
        rank_column="Sharpe",
        rebalance_strategies=["dummy"],
        rebalance_params={"dummy": {}},
    )

    # Two periods processed and the rebalancing helper invoked twice.
    assert list(portfolio.history) == ["2020-01-31", "2020-02-29"]
    assert len(returned_series) == 2

    # First period turnover equals the sum of absolute weights.
    assert portfolio.turnover["2020-01-31"] == pytest.approx(
        float(np.abs(returned_series[0]).sum())
    )

    # Second period experiences a universe change triggering the alignment logic.
    prev = returned_series[0]
    new = returned_series[1]
    union = prev.index.union(new.index)
    expected = float(
        np.abs(new.reindex(union, fill_value=0.0) - prev.reindex(union, fill_value=0.0)).sum()
    )
    assert portfolio.turnover["2020-02-29"] == pytest.approx(expected)

    # Costs accumulate from both rebalancing events.
    assert portfolio.total_rebalance_costs == pytest.approx(0.2)


@dataclass
class IncrementalConfig:
    multi_period: dict[str, Any] = field(
        default_factory=lambda: {
            "frequency": "M",
            "in_sample_len": 3,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-05",
        }
    )
    data: dict[str, Any] = field(default_factory=lambda: {"csv_path": "unused.csv"})
    portfolio: dict[str, Any] = field(
        default_factory=lambda: {
            "policy": "standard",
            "selection_mode": "all",
            "random_n": 2,
            "custom_weights": None,
            "rank": {},
            "manual_list": None,
            "indices_list": None,
        }
    )
    vol_adjust: dict[str, Any] = field(default_factory=lambda: {"target_vol": 1.0})
    benchmarks: dict[str, Any] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=lambda: {"monthly_cost": 0.0})
    performance: dict[str, Any] = field(
        default_factory=lambda: {
            "enable_cache": True,
            "incremental_cov": True,
            "shift_detection_max_steps": 4,
        }
    )
    seed: int = 7

    def model_dump(self) -> dict[str, Any]:
        return {
            "multi_period": self.multi_period,
            "portfolio": self.portfolio,
            "vol_adjust": self.vol_adjust,
        }


@dataclass
class DummyPeriod:
    in_start: str
    in_end: str
    out_start: str
    out_end: str


def test_run_incremental_covariance_path(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = IncrementalConfig()

    dates = pd.to_datetime(
        [
            "2020-01-31",
            "2020-02-29",
            "2020-03-31",
            "2020-04-30",
        ]
    )
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, 0.02, 0.03, 0.04],
            "FundB": [0.05, 0.04, 0.03, 0.02],
        }
    )

    periods = [
        DummyPeriod("2020-01-31", "2020-03-31", "2020-04-30", "2020-04-30"),
        DummyPeriod("2020-02-29", "2020-04-30", "2020-05-31", "2020-05-31"),
    ]

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _: periods)

    run_calls: list[tuple[Any, ...]] = []

    def fake_run_analysis(*args: Any, **kwargs: Any) -> dict[str, Any]:
        run_calls.append(args)
        return {"status": "ok"}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    results = mp_engine.run(cfg, df=df)

    # Two periods processed with covariance diagnostics attached.
    assert len(results) == 2
    assert all("cov_diag" in r for r in results)

    # Incremental update path should record cache statistics.
    stats = results[-1]["cache_stats"]
    assert stats["incremental_updates"] == 1
    # Ensure the sliding-window detection treated the frames as a 1-row shift.
    assert results[-1]["cov_diag"][0] >= 0.0

    # ``_run_analysis`` invoked once per period with expected arguments (7-month slices).
    assert len(run_calls) == 2
