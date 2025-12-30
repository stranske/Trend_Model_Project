"""Additional tests for the multi-period engine incremental covariance
logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import trend_analysis.multi_period.engine as mp_engine


@dataclass
class _Config:
    """Minimal configuration object for exercising ``engine.run``."""

    data: dict[str, Any]
    multi_period: dict[str, Any] = field(
        default_factory=lambda: {"periods": [{"in": {}, "out": {}}]}
    )
    portfolio: dict[str, Any] = field(
        default_factory=lambda: {
            "selection_mode": "all",
            "random_n": 4,
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
        default_factory=lambda: {"enable_cache": True, "incremental_cov": True}
    )
    seed: int = 11

    def model_dump(self) -> dict[str, Any]:
        return {
            "multi_period": self.multi_period,
            "portfolio": self.portfolio,
            "vol_adjust": self.vol_adjust,
        }


@dataclass
class _Period:
    in_start: str
    in_end: str
    out_start: str
    out_end: str


class _DummyCache:
    def __init__(self) -> None:
        self.incremental_updates = 0

    def stats(self) -> dict[str, int]:
        return {
            "entries": 0,
            "hits": 0,
            "misses": 0,
            "incremental_updates": self.incremental_updates,
        }


def _payload_for(frame: pd.DataFrame) -> SimpleNamespace:
    size = frame.shape[1]
    return SimpleNamespace(cov=np.eye(size, dtype=float))


def test_incremental_covariance_falls_back_to_full_recompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no sliding-window shift is detected a full recompute is
    triggered."""

    cfg = _Config(data={"csv_path": "unused.csv"})

    dates = pd.to_datetime(
        [
            "2020-01-31",
            "2020-02-29",
            "2020-03-31",
            "2020-04-30",
            "2020-05-31",
        ]
    )
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, 0.02, 0.03, 0.20, 0.30],
            "FundB": [0.05, 0.04, 0.03, -0.10, -0.20],
        }
    )

    periods: list[_Period] = [
        _Period("2020-01-31", "2020-03-31", "2020-04-30", "2020-04-30"),
        _Period("2020-02-29", "2020-04-30", "2020-05-31", "2020-05-31"),
    ]

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _: periods)

    run_calls: list[tuple[Any, ...]] = []

    def fake_run_analysis(*args: Any, **kwargs: Any) -> dict[str, Any]:
        run_calls.append(args)
        return {"out_ew_stats": {}, "out_user_stats": {}}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    compute_calls: list[int] = []

    def fake_compute_cov_payload(
        frame: pd.DataFrame, *, materialise_aggregates: bool = False
    ) -> SimpleNamespace:
        compute_calls.append(frame.shape[0])
        return _payload_for(frame)

    def fake_incremental_cov_update(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
        raise AssertionError("incremental update path should not be used in fallback test")

    monkeypatch.setattr("trend_analysis.perf.cache.CovCache", _DummyCache)
    monkeypatch.setattr("trend_analysis.perf.cache.compute_cov_payload", fake_compute_cov_payload)
    monkeypatch.setattr(
        "trend_analysis.perf.cache.incremental_cov_update", fake_incremental_cov_update
    )

    results = mp_engine.run(cfg, df=df)

    # run_analysis invoked once per period and covariance recomputed twice
    assert len(run_calls) == len(periods) == 2
    assert compute_calls == [3, 3]

    # Second period falls back to a full recompute without incremental updates
    fallback_stats = results[-1]["cache_stats"]
    assert fallback_stats == {
        "entries": 0,
        "hits": 0,
        "misses": 0,
        "incremental_updates": 0,
    }
    assert results[-1]["cov_diag"] == [1.0, 1.0]
