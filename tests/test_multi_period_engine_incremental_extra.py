from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

import trend_analysis.multi_period.engine as mp_engine


@dataclass
class _Cfg:
    data: Dict[str, Any]
    multi_period: Dict[str, Any]
    portfolio: Dict[str, Any]
    vol_adjust: Dict[str, Any]
    benchmarks: Dict[str, Any]
    run: Dict[str, Any]
    performance: Dict[str, Any]
    seed: int = 5

    def model_dump(self) -> Dict[str, Any]:
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


def _make_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    data = {
        "Date": dates,
        "FundA": np.linspace(0.01, 0.05, len(dates)),
        "FundB": np.linspace(-0.02, 0.02, len(dates)),
    }
    return pd.DataFrame(data)


def test_incremental_update_runs_with_invalid_shift_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _Cfg(
        data={},
        multi_period={"periods": []},
        portfolio={
            "selection_mode": "all",
            "random_n": 4,
            "custom_weights": None,
            "rank": {},
            "manual_list": None,
            "indices_list": None,
        },
        vol_adjust={"target_vol": 1.0},
        benchmarks={},
        run={"monthly_cost": 0.0},
        performance={
            "enable_cache": True,
            "incremental_cov": True,
            "shift_detection_max_steps": "not-an-int",
        },
    )

    periods: List[_Period] = [
        _Period("2020-01-31", "2020-03-31", "2020-04-30", "2020-04-30"),
        _Period("2020-02-29", "2020-04-30", "2020-05-31", "2020-05-31"),
    ]
    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    def fake_run(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"out_ew_stats": {}, "out_user_stats": {}}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run)

    from trend_analysis.perf import cache as perf_cache

    compute_calls: list[int] = []
    incremental_calls: list[int] = []

    original_compute = perf_cache.compute_cov_payload
    original_incremental = perf_cache.incremental_cov_update

    def tracking_compute(frame: pd.DataFrame, *, materialise_aggregates: bool) -> Any:
        compute_calls.append(frame.shape[0])
        return original_compute(frame, materialise_aggregates=materialise_aggregates)

    def tracking_incremental(prev: Any, old_row: np.ndarray, new_row: np.ndarray) -> Any:
        incremental_calls.append(1)
        return original_incremental(prev, old_row, new_row)

    monkeypatch.setattr("trend_analysis.perf.cache.compute_cov_payload", tracking_compute)
    monkeypatch.setattr("trend_analysis.perf.cache.incremental_cov_update", tracking_incremental)

    results = mp_engine.run(cfg, df=_make_df())

    assert len(results) == len(periods)
    # First window uses a full compute; second window applies incremental update only
    assert compute_calls == [3]
    assert incremental_calls == [1]
    assert "cov_diag" in results[-1]
    assert results[-1]["cache_stats"]["incremental_updates"] == 1


def test_incremental_update_fallback_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _Cfg(
        data={},
        multi_period={"periods": []},
        portfolio={
            "selection_mode": "all",
            "random_n": 4,
            "custom_weights": None,
            "rank": {},
            "manual_list": None,
            "indices_list": None,
        },
        vol_adjust={"target_vol": 1.0},
        benchmarks={},
        run={"monthly_cost": 0.0},
        performance={"enable_cache": True, "incremental_cov": True},
    )

    periods: List[_Period] = [
        _Period("2020-01-31", "2020-03-31", "2020-04-30", "2020-04-30"),
        _Period("2020-02-29", "2020-04-30", "2020-05-31", "2020-05-31"),
    ]
    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    def fake_run(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"out_ew_stats": {}, "out_user_stats": {}}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run)

    from trend_analysis.perf import cache as perf_cache

    compute_calls: list[int] = []
    original_compute = perf_cache.compute_cov_payload
    monkeypatch.setattr(
        "trend_analysis.perf.cache.compute_cov_payload",
        lambda frame, *, materialise_aggregates: (
            compute_calls.append(frame.shape[0])
            or original_compute(frame, materialise_aggregates=materialise_aggregates)
        ),
    )

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr("trend_analysis.perf.cache.incremental_cov_update", boom)

    results = mp_engine.run(cfg, df=_make_df())

    assert len(results) == len(periods)
    # First window uses a full compute; second falls back to full recompute after failure.
    assert compute_calls == [3, 3]
    assert results[-1]["cache_stats"]["incremental_updates"] == 0
    assert "cov_diag" in results[-1]


def test_incremental_update_length_change_triggers_recompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _Cfg(
        data={},
        multi_period={"periods": []},
        portfolio={
            "selection_mode": "all",
            "random_n": 4,
            "custom_weights": None,
            "rank": {},
            "manual_list": None,
            "indices_list": None,
        },
        vol_adjust={"target_vol": 1.0},
        benchmarks={},
        run={"monthly_cost": 0.0},
        performance={"enable_cache": True, "incremental_cov": True},
    )

    periods: List[_Period] = [
        _Period("2020-01-31", "2020-03-31", "2020-04-30", "2020-04-30"),
        _Period("2020-02-29", "2020-05-31", "2020-06-30", "2020-06-30"),
    ]
    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    def fake_run(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"out_ew_stats": {}, "out_user_stats": {}}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run)

    from trend_analysis.perf import cache as perf_cache

    compute_calls: list[int] = []
    original_compute = perf_cache.compute_cov_payload

    def tracking_compute(frame: pd.DataFrame, *, materialise_aggregates: bool) -> Any:
        compute_calls.append(frame.shape[0])
        return original_compute(frame, materialise_aggregates=materialise_aggregates)

    monkeypatch.setattr("trend_analysis.perf.cache.compute_cov_payload", tracking_compute)

    results = mp_engine.run(cfg, df=_make_df())

    assert len(results) == len(periods)
    # Different window lengths should trigger a fresh covariance computation.
    assert compute_calls == [3, 4]
    assert results[-1]["cache_stats"]["incremental_updates"] == 0
    assert "cov_diag" in results[-1]


def test_incremental_cov_cache_instantiation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _Cfg(
        data={},
        multi_period={"periods": []},
        portfolio={
            "selection_mode": "all",
            "random_n": 4,
            "custom_weights": None,
            "rank": {},
            "manual_list": None,
            "indices_list": None,
        },
        vol_adjust={"target_vol": 1.0},
        benchmarks={},
        run={"monthly_cost": 0.0},
        performance={"enable_cache": True, "incremental_cov": True},
    )

    periods: List[_Period] = [
        _Period("2020-01-31", "2020-03-31", "2020-04-30", "2020-04-30"),
    ]
    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    def fake_run(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        return {"out_ew_stats": {}, "out_user_stats": {}}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run)

    class ExplodingCovCache:
        def __init__(self) -> None:
            raise RuntimeError("no cache available")

    monkeypatch.setattr("trend_analysis.perf.cache.CovCache", ExplodingCovCache)

    results = mp_engine.run(cfg, df=_make_df())

    assert len(results) == 1
    payload = results[0]
    # Cache instantiation failure should not attach cache_stats metadata
    assert "cache_stats" not in payload
    assert "cov_diag" in payload
