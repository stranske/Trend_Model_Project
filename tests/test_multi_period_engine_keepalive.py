"""Coverage-focused tests for ``multi_period.engine`` keepalive gaps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd
import pytest

from trend_analysis.multi_period import engine as mp_engine
from trend_analysis.weighting import BaseWeighting


@dataclass
class BasicConfig:
    """Minimal configuration object compatible with ``mp_engine.run``."""

    multi_period: Dict[str, Any] = field(
        default_factory=lambda: {
            "frequency": "M",
            "in_sample_len": 1,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-02",
        }
    )
    data: Dict[str, Any] = field(default_factory=lambda: {"csv_path": "unused.csv"})
    portfolio: Dict[str, Any] = field(
        default_factory=lambda: {
            "policy": "standard",
            "selection_mode": "all",
            "random_n": 2,
            "custom_weights": None,
            "rank": {},
            "manual_list": None,
            "indices_list": None,
            "weighting": {"name": "equal", "params": {}},
        }
    )
    vol_adjust: Dict[str, Any] = field(default_factory=lambda: {"target_vol": 1.0})
    benchmarks: Dict[str, Any] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=lambda: {"monthly_cost": 0.0})
    seed: int = 42
    performance: Dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> Dict[str, Any]:
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


class StaticSelector:
    """Selector that preserves ordering and exposes ``rank_column``."""

    rank_column = "Sharpe"

    def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return score_frame, score_frame


class EmptyRebalancer:
    """Rebalancer that forces an empty universe to exercise reseed branches."""

    def __init__(self, *_cfg: Any) -> None:
        self.calls = 0

    def apply_triggers(self, prev_weights: pd.Series, _sf: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        self.calls += 1
        return prev_weights.iloc[0:0]


def _patch_generate_periods(monkeypatch: pytest.MonkeyPatch, periods: List[DummyPeriod]) -> None:
    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)


def test_run_schedule_handles_missing_rank_column(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = {
        "2024-01-31": pd.DataFrame({"Sharpe": [1.0], "Other": [0.5]}, index=["FundA"]),
        "2024-02-29": pd.DataFrame({"Sharpe": [0.9], "Other": [0.4]}, index=["FundA"]),
    }

    class Selector:
        def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            return score_frame, score_frame

    class Weighting(BaseWeighting):
        def __init__(self) -> None:
            self.update_calls: list[int] = []

        def weight(self, selected: pd.DataFrame, date: pd.Timestamp | None = None) -> pd.DataFrame:
            del date
            return pd.DataFrame({"weight": [1.0]}, index=selected.index)

        def update(self, scores: pd.Series, days: int) -> None:  # pragma: no cover - invoked conditionally
            self.update_calls.append(days)

    weighting = Weighting()

    portfolio = mp_engine.run_schedule(
        frames,
        Selector(),
        weighting,
        rank_column="Missing",
    )

    assert portfolio.history
    assert not weighting.update_calls


def test_run_uses_nan_policy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = BasicConfig()
    cfg.performance = {"enable_cache": False}
    cfg.data.update({"nan_policy": "bfill", "nan_limit": 7})

    captured: dict[str, Any] = {}

    def fake_missing_policy(frame: pd.DataFrame, *, policy: str, limit: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        captured["policy"] = policy
        captured["limit"] = limit
        return frame, {"applied": True}

    dates = ["2020-01-31", "2020-02-29", "2020-03-31"]
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.1, 0.2, 0.3],
            "FundB": [0.05, 0.06, 0.07],
        }
    )

    _patch_generate_periods(
        monkeypatch,
        [DummyPeriod("2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29")],
    )
    monkeypatch.setattr(mp_engine, "apply_missing_policy", fake_missing_policy)
    monkeypatch.setattr(mp_engine, "_run_analysis", lambda *_args, **_kwargs: {"summary": "ok"})

    results = mp_engine.run(cfg, df=df)

    assert captured["policy"] == "bfill"
    assert captured["limit"] == 7
    assert results and results[0]["period"] == (
        "2020-01-31",
        "2020-01-31",
        "2020-02-29",
        "2020-02-29",
    )


def test_run_skips_missing_policy_when_price_frames_present(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = BasicConfig()
    cfg.performance = {"enable_cache": False}

    frame_one = pd.DataFrame(
        {
            "Date": ["2020-01-31", "2020-02-29"],
            "FundA": [0.1, 0.2],
        }
    )
    frame_two = pd.DataFrame(
        {
            "Date": ["2020-02-29", "2020-03-31"],
            "FundB": [0.3, 0.4],
        }
    )

    called = False

    def fail_missing_policy(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - guard
        nonlocal called
        called = True
        raise AssertionError("apply_missing_policy should not be invoked")

    captures: list[pd.DataFrame] = []

    def fake_run_analysis(df: pd.DataFrame, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        captures.append(df.copy())
        return {"analysis": "ok"}

    _patch_generate_periods(
        monkeypatch,
        [DummyPeriod("2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29")],
    )
    monkeypatch.setattr(mp_engine, "apply_missing_policy", fail_missing_policy)
    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    results = mp_engine.run(
        cfg,
        price_frames={
            "2020-01-31": frame_one,
            "2020-02-29": frame_two,
        },
    )

    assert not called
    assert results
    combined = captures[0]
    assert list(combined["Date"]) == list(pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]))
    assert set(combined.columns) == {"Date", "FundA", "FundB"}


def test_run_raises_when_loader_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = BasicConfig()
    cfg.data = {"csv_path": "missing.csv"}

    monkeypatch.setattr(mp_engine, "load_csv", lambda *_a, **_k: None)

    with pytest.raises(ValueError, match="Failed to load CSV data"):
        mp_engine.run(cfg, df=None)


def test_threshold_hold_returns_placeholder_for_empty_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = BasicConfig()
    cfg.portfolio = {
        "policy": "threshold_hold",
        "threshold_hold": {"target_n": 2, "metric": "Sharpe"},
        "constraints": {"max_funds": 2, "min_weight": 0.05, "max_weight": 0.9},
        "weighting": {"name": "equal", "params": {}},
        "indices_list": None,
        "random_n": 2,
    }
    cfg.performance = {"enable_cache": False}

    dates = pd.to_datetime(["2020-01-31", "2020-02-29"])
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.1, float("nan")],
            "FundB": [float("nan"), 0.2],
        }
    )

    _patch_generate_periods(
        monkeypatch,
        [DummyPeriod("2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29")],
    )
    monkeypatch.setattr(
        "trend_analysis.selector.create_selector_by_name",
        lambda *_a, **_k: StaticSelector(),
    )
    monkeypatch.setattr(mp_engine, "Rebalancer", EmptyRebalancer)
    monkeypatch.setattr(mp_engine, "_run_analysis", lambda *_a, **_k: {"payload": "unused"})

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 1
    entry = results[0]
    assert entry["selected_funds"] == []
    assert entry["in_sample_scaled"].empty
    assert entry["manager_changes"] == []


def test_run_loads_csv_with_nan_policy_and_string_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = BasicConfig()
    cfg.performance = {"enable_cache": False}
    cfg.data.update({"nan_policy": "ffill", "nan_limit": 3})

    loaded = pd.DataFrame(
        {
            "Date": ["2020-01-31", "2020-02-29"],
            "FundA": [0.1, 0.2],
        }
    )

    _patch_generate_periods(
        monkeypatch,
        [DummyPeriod("2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29")],
    )

    monkeypatch.setattr(mp_engine, "load_csv", lambda *_a, **_k: loaded.copy())

    captured_dates: list[pd.Timestamp] = []

    def fake_run_analysis(df: pd.DataFrame, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        captured_dates.extend(df["Date"].tolist())
        return {"ok": True}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    results = mp_engine.run(cfg, df=None)

    assert results
    assert captured_dates == list(pd.to_datetime(["2020-01-31", "2020-02-29"]))


