from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import pytest

import trend_analysis.multi_period.engine as mp_engine
import trend_analysis.selector as selector_mod


@dataclass
class MinimalConfig:
    """Configuration focused on exercising branch-heavy paths."""

    multi_period: Dict[str, Any] = field(
        default_factory=lambda: {
            "frequency": "M",
            "in_sample_len": 2,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-03",
        }
    )
    data: Dict[str, Any] = field(default_factory=lambda: {"csv_path": "unused.csv"})
    portfolio: Dict[str, Any] = field(
        default_factory=lambda: {
            "policy": "threshold_hold",
            "random_n": 4,
            "transaction_cost_bps": 0.0,
            "max_turnover": 1.0,
            "threshold_hold": {"target_n": 3, "metric": "Sharpe"},
            "constraints": {
                "max_funds": 1,
                "min_weight": 0.05,
                "max_weight": 0.6,
                "min_weight_strikes": 1,
            },
            "weighting": {"name": "mystery", "params": {}},
            "indices_list": None,
        }
    )
    vol_adjust: Dict[str, Any] = field(default_factory=lambda: {"target_vol": 1.0})
    benchmarks: Dict[str, Any] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=lambda: {"monthly_cost": 0.0})
    seed: int = 99

    def model_dump(self) -> Dict[str, Any]:
        return {
            "multi_period": self.multi_period,
            "portfolio": self.portfolio,
            "vol_adjust": self.vol_adjust,
        }


class SelectorStub:
    def __init__(self, ordering: Iterable[str]) -> None:
        self._ordering = list(ordering)

    def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        ordered = [ix for ix in self._ordering if ix in score_frame.index]
        selected = score_frame.loc[ordered]
        return selected, selected


class WeightingStub:
    """Return intentionally oversized weights to exercise bound logic."""

    def __init__(self) -> None:
        self.updates: list[tuple[pd.Series, int]] = []

    def weight(
        self, selected: pd.DataFrame, date: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        del date
        base = [0.7, 0.6, 0.5]
        values = pd.Series(base[: len(selected)], index=selected.index, dtype=float)
        return values.to_frame("weight")

    def update(self, scores: pd.Series, days: int) -> None:
        self.updates.append((scores.astype(float), int(days)))


class IdentityRebalancer:
    def __init__(self, *_: Any) -> None:
        pass

    def apply_triggers(self, prev_weights: pd.Series, _: pd.DataFrame) -> pd.Series:
        return prev_weights


@dataclass
class PerPeriodConfig:
    """Configuration that exercises the non-threshold multi-period path."""

    multi_period: Dict[str, Any] = field(
        default_factory=lambda: {
            "frequency": "M",
            "in_sample_len": 1,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-02",
        }
    )
    data: Dict[str, Any] = field(default_factory=lambda: {"csv_path": "dummy.csv"})
    portfolio: Dict[str, Any] = field(
        default_factory=lambda: {
            "policy": "random",
            "selection_mode": "all",
            "random_n": 2,
            "custom_weights": None,
            "rank": None,
            "manual_list": None,
            "indices_list": None,
            "previous_weights": None,
            "max_turnover": 1.0,
        }
    )
    vol_adjust: Dict[str, Any] = field(default_factory=lambda: {"target_vol": 1.0})
    performance: Dict[str, Any] = field(
        default_factory=lambda: {"enable_cache": True, "incremental_cov": True}
    )
    benchmarks: Dict[str, Any] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=lambda: {"monthly_cost": 0.0})
    seed: int = 7

    def model_dump(self) -> Dict[str, Any]:
        return {
            "multi_period": self.multi_period,
            "portfolio": self.portfolio,
            "vol_adjust": self.vol_adjust,
        }


def _stub_result(period: Any) -> Dict[str, Any]:
    return {
        "period": (
            period.in_start,
            period.in_end,
            period.out_start,
            period.out_end,
        ),
        "selected_funds": ["FundA"],
        "in_sample_scaled": pd.DataFrame(),
        "out_sample_scaled": pd.DataFrame(),
        "in_sample_stats": {},
        "out_sample_stats": {},
        "out_sample_stats_raw": {},
        "in_ew_stats": (),
        "out_ew_stats": (),
        "out_ew_stats_raw": (),
        "in_user_stats": (),
        "out_user_stats": (),
        "out_user_stats_raw": (),
        "ew_weights": {},
        "fund_weights": {},
        "benchmark_stats": {},
        "benchmark_ir": {},
        "score_frame": pd.DataFrame(),
        "weight_engine_fallback": None,
    }


def _stub_run_analysis(*_: Any, **__: Any) -> Dict[str, Any]:
    return {
        "selected_funds": ["FundA"],
        "in_sample_scaled": pd.DataFrame(),
        "out_sample_scaled": pd.DataFrame(),
        "in_sample_stats": {},
        "out_sample_stats": {},
        "out_sample_stats_raw": {},
        "in_ew_stats": (),
        "out_ew_stats": (),
        "out_ew_stats_raw": (),
        "in_user_stats": (),
        "out_user_stats": (),
        "out_user_stats_raw": (),
        "ew_weights": {},
        "fund_weights": {},
        "benchmark_stats": {},
        "benchmark_ir": {},
        "score_frame": pd.DataFrame(),
        "weight_engine_fallback": None,
    }


def test_run_covers_threshold_hold_branches(monkeypatch):
    cfg = MinimalConfig()
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "FundA": [0.01, 0.02, 0.03],
            "FundB": [0.015, 0.0, 0.01],
            "FundC": [0.02, 0.01, 0.0],
        }
    )

    periods = [
        SimpleNamespace(
            in_start="2020-01",
            in_end="2020-02",
            out_start="2020-03",
            out_end="2020-03",
        )
    ]

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _: periods)
    monkeypatch.setattr(mp_engine, "_run_analysis", _stub_run_analysis)
    monkeypatch.setattr(
        selector_mod,
        "create_selector_by_name",
        lambda *_, **__: SelectorStub(["FundA", "FundB", "FundC"]),
    )
    monkeypatch.setattr(mp_engine, "EqualWeight", lambda: WeightingStub())
    monkeypatch.setattr(mp_engine, "Rebalancer", lambda *_: IdentityRebalancer())

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 1
    entry = results[0]
    assert entry["manager_changes"]  # events recorded from branch coverage


def test_run_schedule_calls_weight_update() -> None:
    frame = pd.DataFrame({"Sharpe": [1.2], "Other": [0.0]}, index=["FundA"])

    class UpdateWeighting(WeightingStub):
        def weight(self, selected: pd.DataFrame, *_: Any, **__: Any) -> pd.DataFrame:
            return selected[["Sharpe"]].rename(columns={"Sharpe": "weight"})

    selector = SelectorStub(["FundA"])
    weighting = UpdateWeighting()

    mp_engine.run_schedule(
        {"2020-01-31": frame}, selector, weighting, rank_column="Sharpe"
    )

    assert weighting.updates and weighting.updates[0][1] == 0


def test_run_loads_csv_and_handles_missing_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = PerPeriodConfig()

    raw_df = pd.DataFrame(
        {
            "Date": ["2020-01-31", "2020-02-29"],
            "FundA": [0.01, 0.02],
            "FundB": [0.02, 0.03],
        }
    )

    monkeypatch.setattr(mp_engine, "load_csv", lambda *a, **k: raw_df.copy())

    periods = [
        SimpleNamespace(
            in_start="2020-01",
            in_end="2020-01",
            out_start="2020-02",
            out_end="2020-02",
        ),
        SimpleNamespace(
            in_start="2020-02",
            in_end="2020-02",
            out_start="2020-03",
            out_end="2020-03",
        ),
    ]
    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    results_iter = iter([None, _stub_result(periods[1])])

    def fake_run_analysis(*args: Any, **kwargs: Any) -> Dict[str, Any] | None:
        return next(results_iter)

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    import trend_analysis.perf.cache as cache_mod

    monkeypatch.setattr(
        cache_mod,
        "CovCache",
        lambda: SimpleNamespace(stats=lambda: {"hits": 0}, incremental_updates=0),
    )

    class DummyCovPayload:
        def __init__(self, size: int) -> None:
            self.cov = np.eye(size)

    monkeypatch.setattr(
        cache_mod,
        "compute_cov_payload",
        lambda df, materialise_aggregates=False: DummyCovPayload(df.shape[1]),
    )
    monkeypatch.setattr(
        cache_mod,
        "incremental_cov_update",
        lambda payload, old, new: payload,
    )

    output = mp_engine.run(cfg, df=None)

    assert len(output) == 1


def test_run_load_csv_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = PerPeriodConfig()
    monkeypatch.setattr(mp_engine, "load_csv", lambda *a, **k: None)
    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: [])

    with pytest.raises(ValueError):
        mp_engine.run(cfg, df=None)


def test_run_price_frames_skip_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = MinimalConfig()
    cfg.portfolio["policy"] = "threshold_hold"
    cfg.portfolio["threshold_hold"]["target_n"] = 1

    price_frames = {
        "first": pd.DataFrame(
            {
                "Date": ["2020-01-31", "2020-02-29"],
                "FundA": [0.01, 0.02],
                "FundB": [0.02, 0.03],
            }
        )
    }

    periods = [
        SimpleNamespace(
            in_start="2020-01",
            in_end="2020-01",
            out_start="2020-03",
            out_end="2020-03",
        )
    ]

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)
    monkeypatch.setattr(mp_engine, "_run_analysis", _stub_run_analysis)
    monkeypatch.setattr(
        selector_mod,
        "create_selector_by_name",
        lambda *_, **__: SelectorStub(["FundA", "FundB"]),
    )
    monkeypatch.setattr(mp_engine, "EqualWeight", lambda: WeightingStub())
    monkeypatch.setattr(mp_engine, "Rebalancer", lambda *_: IdentityRebalancer())

    results = mp_engine.run(cfg, df=None, price_frames=price_frames)

    assert len(results) == 1
    assert results[0]["selected_funds"] == []
