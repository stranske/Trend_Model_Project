"""Tests covering the incremental covariance path in the multi-period
engine."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from trend_analysis.multi_period import engine as mp_engine
from trend_analysis.perf import cache as cache_mod


class _Cfg:
    """Minimal configuration object for exercising ``engine.run``."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.portfolio: dict[str, object] = {
            "policy": "",  # ensure Phase-1 style path is used
            "selection_mode": "all",
            "random_n": 2,
            "rank": {},
            "custom_weights": None,
            "manual_list": None,
            "indices_list": None,
        }
        self.vol_adjust: dict[str, float] = {"target_vol": 1.0}
        self.run: dict[str, float] = {"monthly_cost": 0.0}
        self.benchmarks: dict[str, str] = {}
        self.performance: dict[str, object] = {
            "enable_cache": True,
            "incremental_cov": True,
            "shift_detection_max_steps": 4,
        }
        self.seed = 7

    def model_dump(self) -> dict[str, object]:
        return {}


def _make_df() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2020-01-31",
            "2020-02-29",
            "2020-03-31",
            "2020-04-30",
            "2020-05-31",
        ]
    )
    return pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, 0.015, 0.012, 0.011, 0.013],
            "FundB": [0.005, 0.007, 0.006, 0.008, 0.009],
        }
    )


def _make_periods() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            in_start="2020-01",
            in_end="2020-03",
            out_start="2020-04",
            out_end="2020-04",
        ),
        SimpleNamespace(
            in_start="2020-02",
            in_end="2020-04",
            out_start="2020-05",
            out_end="2020-05",
        ),
    ]


def test_run_incremental_covariance_updates(monkeypatch):
    """Exercise the shift-detection incremental covariance path."""

    cfg = _Cfg()
    df = _make_df()
    periods = _make_periods()

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)
    monkeypatch.setattr(mp_engine.np, "allclose", lambda *a, **kwargs: False)

    run_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_run_analysis(*args, **kwargs):
        run_calls.append((args, kwargs))
        return {"out_ew_stats": {"sharpe": 1.0}}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    real_incremental = cache_mod.incremental_cov_update
    incremental_calls: list[tuple[pd.Series, pd.Series]] = []

    def wrapped_incremental(prev, old_row, new_row):
        incremental_calls.append((pd.Series(old_row), pd.Series(new_row)))
        return real_incremental(prev, old_row, new_row)

    monkeypatch.setattr(cache_mod, "incremental_cov_update", wrapped_incremental)

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2
    assert len(run_calls) == 2
    # Second period should rely on incremental updates (k == 1)
    assert len(incremental_calls) == 1
    second_stats = results[1]["cache_stats"]
    assert second_stats["incremental_updates"] == 1


@pytest.mark.parametrize("non_positive_value", [0, -1])
def test_run_incremental_covariance_coerces_non_positive_shift_steps(
    monkeypatch, non_positive_value
):
    """Non-positive shift step settings should coerce to at least one step."""

    cfg = _Cfg()
    cfg.performance["shift_detection_max_steps"] = non_positive_value  # force coercion branch
    df = _make_df()
    periods = _make_periods()

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)
    monkeypatch.setattr(mp_engine.np, "allclose", lambda *a, **kwargs: False)

    def fake_run_analysis(*args, **kwargs):
        return {"out_ew_stats": {"sharpe": 1.0}}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    real_incremental = cache_mod.incremental_cov_update
    incremental_calls: list[int] = []

    def wrapped_incremental(prev, old_row, new_row):
        incremental_calls.append(1)
        return real_incremental(prev, old_row, new_row)

    monkeypatch.setattr(cache_mod, "incremental_cov_update", wrapped_incremental)

    real_compute = cache_mod.compute_cov_payload
    compute_calls: list[int] = []

    def wrapped_compute(frame: pd.DataFrame, *, materialise_aggregates: bool):
        compute_calls.append(frame.shape[0])
        return real_compute(frame, materialise_aggregates=materialise_aggregates)

    monkeypatch.setattr(cache_mod, "compute_cov_payload", wrapped_compute)

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2
    # Coercing to one step still results in a full recomputation when the limit is non-positive.
    assert incremental_calls == []
    assert compute_calls == [3, 3]
    assert results[1]["cache_stats"]["incremental_updates"] == 0


def test_run_incremental_covariance_shift_detection(monkeypatch):
    """Ensure the incremental path applies sequential updates when shifts are
    detected."""

    cfg = _Cfg()
    df = _make_df()
    periods = _make_periods()

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    run_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_run_analysis(*args, **kwargs):
        run_calls.append((args, kwargs))
        return {"out_ew_stats": {"sharpe": 1.0}}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2
    assert len(run_calls) == 2
    second_stats = results[1]["cache_stats"]
    # Expect a single incremental update from the detected one-row shift.
    assert second_stats["incremental_updates"] == 1


def test_run_incremental_covariance_shift_detection_via_allclose(monkeypatch):
    """Shift detection should succeed when ``np.allclose`` alone finds the
    overlap."""

    cfg = _Cfg()
    df = _make_df()
    periods = _make_periods()

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    def fake_run_analysis(*args, **kwargs):
        return {"out_ew_stats": {"sharpe": 1.0}}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    real_incremental = cache_mod.incremental_cov_update
    incremental_calls: list[int] = []

    def tracking_incremental(prev, old_row, new_row):
        incremental_calls.append(1)
        return real_incremental(prev, old_row, new_row)

    monkeypatch.setattr(cache_mod, "incremental_cov_update", tracking_incremental)

    call_counter = {"calls": 0}

    def tracking_allclose(*args, **kwargs):
        call_counter["calls"] += 1
        return True

    monkeypatch.setattr(mp_engine.np, "allclose", tracking_allclose)

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2
    assert incremental_calls == [1]
    assert call_counter["calls"] >= 1


def test_run_incremental_covariance_fallback_on_error(monkeypatch):
    """If incremental updates fail, the engine recomputes the covariance."""

    cfg = _Cfg()
    df = _make_df()
    periods = _make_periods()

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    def fake_run_analysis(*args, **kwargs):
        return {"out_ew_stats": {"sharpe": 1.0}}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    fallback_calls: list[int] = []
    real_compute = cache_mod.compute_cov_payload

    def wrapped_compute(df: pd.DataFrame, *, materialise_aggregates: bool):
        fallback_calls.append(df.shape[0])
        return real_compute(df, materialise_aggregates=materialise_aggregates)

    monkeypatch.setattr(cache_mod, "compute_cov_payload", wrapped_compute)

    def boom(prev, old_row, new_row):
        raise RuntimeError("incremental update failure")

    monkeypatch.setattr(cache_mod, "incremental_cov_update", boom)

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2
    # compute_cov_payload is invoked once per period; failure triggers a second call for the second period
    assert fallback_calls == [3, 3]
    second_stats = results[1]["cache_stats"]
    # No incremental updates recorded because the helper kept failing
    assert second_stats["incremental_updates"] == 0


def test_run_incremental_covariance_handles_bad_shift_and_strings(monkeypatch):
    """Test that the engine correctly handles string-formatted dates and
    invalid (non-integer) shift detection settings.

    Ensures that covariance computation and diagnostics are performed
    even when input formats are incorrect or edge cases are present.
    """
    cfg = _Cfg()
    cfg.performance["shift_detection_max_steps"] = "not-an-int"
    cfg.benchmarks = {"bm": "Benchmark"}
    cfg.portfolio["indices_list"] = ["Benchmark"]

    df = _make_df()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["Benchmark"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    periods = _make_periods()

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    compute_calls: list[int] = []
    real_compute = cache_mod.compute_cov_payload

    def wrapped_compute(frame: pd.DataFrame, *, materialise_aggregates: bool):
        compute_calls.append(frame.shape[0])
        return real_compute(frame, materialise_aggregates=materialise_aggregates)

    monkeypatch.setattr(cache_mod, "compute_cov_payload", wrapped_compute)

    def fake_run_analysis(*args, **kwargs):
        return {"out_ew_stats": {"sharpe": 1.0}}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2
    # Invalid shift thresholds still compute covariance and expose diagnostics
    assert compute_calls[0] == 3
    assert all("cov_diag" in result for result in results)


def test_run_incremental_covariance_multi_step_update(monkeypatch):
    """Incremental covariance handles multi-row shifts via sequential
    updates."""

    cfg = _Cfg()
    dates = pd.to_datetime(
        [
            "2020-01-31",
            "2020-02-29",
            "2020-03-31",
            "2020-04-30",
            "2020-05-31",
            "2020-06-30",
        ]
    )
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, 0.011, 0.012, 0.013, 0.014, 0.015],
            "FundB": [0.005, 0.006, 0.007, 0.008, 0.009, 0.010],
        }
    )
    periods = [
        SimpleNamespace(
            in_start="2020-01",
            in_end="2020-04",
            out_start="2020-05",
            out_end="2020-05",
        ),
        SimpleNamespace(
            in_start="2020-03",
            in_end="2020-06",
            out_start="2020-06",
            out_end="2020-06",
        ),
    ]

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    def fake_run_analysis(*args, **kwargs):
        return {"out_ew_stats": {"sharpe": 1.0}}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    real_incremental = cache_mod.incremental_cov_update
    incremental_calls: list[tuple[pd.Series, pd.Series]] = []

    def wrapped_incremental(prev, old_row, new_row):
        incremental_calls.append((pd.Series(old_row), pd.Series(new_row)))
        return real_incremental(prev, old_row, new_row)

    monkeypatch.setattr(cache_mod, "incremental_cov_update", wrapped_incremental)

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2
    assert len(incremental_calls) == 2
    first_old, first_new = incremental_calls[0]
    second_old, second_new = incremental_calls[1]
    # Old rows roll off in chronological order while new data arrives at the tail.
    assert list(first_old.index) == [0, 1]
    assert list(second_old.index) == [0, 1]
    assert first_new.iloc[0] == pytest.approx(0.014)
    assert second_new.iloc[0] == pytest.approx(0.015)
    stats = results[1]["cache_stats"]
    assert stats["incremental_updates"] == 2


def test_run_incremental_covariance_shift_detection_no_match(monkeypatch):
    """If no shift is detected within the configured window, fall back to a
    full covariance recomputation."""

    cfg = _Cfg()
    cfg.performance["shift_detection_max_steps"] = 1
    df = _make_df()
    periods = [
        SimpleNamespace(
            in_start="2020-01",
            in_end="2020-03",
            out_start="2020-04",
            out_end="2020-04",
        ),
        SimpleNamespace(
            in_start="2020-03",
            in_end="2020-05",
            out_start="2020-05",
            out_end="2020-05",
        ),
    ]

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    def fake_run_analysis(*args, **kwargs):
        return {"out_ew_stats": {"sharpe": 1.0}}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    real_compute = cache_mod.compute_cov_payload
    compute_calls: list[int] = []

    def wrapped_compute(frame: pd.DataFrame, *, materialise_aggregates: bool):
        compute_calls.append(frame.shape[0])
        return real_compute(frame, materialise_aggregates=materialise_aggregates)

    monkeypatch.setattr(cache_mod, "compute_cov_payload", wrapped_compute)

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2
    # Initial computation plus fallback recompute for the second period.
    assert compute_calls == [3, 3]


def test_run_incremental_covariance_handles_allclose_failure(monkeypatch):
    """Shift detection should fall back to ``np.array_equal`` if
    ``np.allclose`` fails."""

    cfg = _Cfg()
    df = _make_df()
    periods = _make_periods()

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    def fake_run_analysis(*args, **kwargs):
        return {"out_ew_stats": {"sharpe": 1.0}}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    calls: list[str] = []

    def tracking_allclose(*args, **kwargs):
        calls.append("allclose")
        return False

    def tracking_array_equal(*args, **kwargs):
        calls.append("array_equal")
        return True

    monkeypatch.setattr(mp_engine.np, "allclose", tracking_allclose)
    monkeypatch.setattr(mp_engine.np, "array_equal", tracking_array_equal)

    real_incremental = cache_mod.incremental_cov_update
    incremental_calls: list[int] = []

    def wrapped_incremental(prev, old_row, new_row):
        incremental_calls.append(1)
        return real_incremental(prev, old_row, new_row)

    monkeypatch.setattr(cache_mod, "incremental_cov_update", wrapped_incremental)

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2
    assert incremental_calls == [1]
    assert calls.count("array_equal") >= 1


def test_run_incremental_covariance_column_change_triggers_recompute(monkeypatch):
    """Changing the column universe forces a fresh covariance computation."""

    cfg = _Cfg()
    df = _make_df().copy()
    df["FundC"] = [0.003, 0.004, 0.002, 0.001, 0.0]
    periods = _make_periods()

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    def fake_run_analysis(*args, **kwargs):
        return {"out_ew_stats": {"sharpe": 1.0}}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    original_prepare = mp_engine._prepare_returns_frame
    call_count = {"n": 0}

    def dropping_prepare(frame: pd.DataFrame) -> pd.DataFrame:
        call_count["n"] += 1
        prepared = original_prepare(frame)
        if call_count["n"] == 2 and "FundC" in prepared.columns:
            return prepared.drop(columns=["FundC"])
        return prepared

    monkeypatch.setattr(mp_engine, "_prepare_returns_frame", dropping_prepare)

    real_compute = cache_mod.compute_cov_payload
    compute_calls: list[int] = []

    def tracking_compute(frame: pd.DataFrame, *, materialise_aggregates: bool):
        compute_calls.append(frame.shape[1])
        return real_compute(frame, materialise_aggregates=materialise_aggregates)

    monkeypatch.setattr(cache_mod, "compute_cov_payload", tracking_compute)

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2
    # Initial computation plus forced recompute after the column drop.
    assert compute_calls[0] == 3
    assert compute_calls[1] == 2
    stats = results[1]["cache_stats"]
    assert stats["incremental_updates"] == 0


def test_run_incremental_covariance_longer_window_forces_full_recompute(
    monkeypatch,
) -> None:
    """When the in-sample window length changes, the engine recomputes
    covariance."""

    cfg = _Cfg()
    df = _make_df()
    periods = [
        SimpleNamespace(
            in_start="2020-01",
            in_end="2020-03",
            out_start="2020-04",
            out_end="2020-04",
        ),
        SimpleNamespace(
            in_start="2020-01",
            in_end="2020-04",
            out_start="2020-05",
            out_end="2020-05",
        ),
    ]

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    def fake_run_analysis(*args, **kwargs):
        return {"out_ew_stats": {"sharpe": 1.0}}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    real_compute = cache_mod.compute_cov_payload
    compute_calls: list[int] = []

    def tracking_compute(frame: pd.DataFrame, *, materialise_aggregates: bool):
        compute_calls.append(frame.shape[0])
        return real_compute(frame, materialise_aggregates=materialise_aggregates)

    monkeypatch.setattr(cache_mod, "compute_cov_payload", tracking_compute)

    incremental_calls: list[int] = []
    real_incremental = cache_mod.incremental_cov_update

    def tracking_incremental(prev, old_row, new_row):
        incremental_calls.append(1)
        return real_incremental(prev, old_row, new_row)

    monkeypatch.setattr(cache_mod, "incremental_cov_update", tracking_incremental)

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2
    # First period computes covariance for 3 rows, second recomputes for 4 rows.
    assert compute_calls == [3, 4]
    assert not incremental_calls


def test_run_incremental_covariance_handles_cov_cache_import_failure(monkeypatch):
    """If CovCache cannot be imported the engine should proceed without cache
    stats."""

    cfg = _Cfg()
    df = _make_df()
    periods = _make_periods()

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    def fake_run_analysis(*args, **kwargs):
        return {"out_ew_stats": {"sharpe": 1.0}}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    # Patch CovCache to raise ImportError when accessed
    monkeypatch.setattr(
        cache_mod,
        "CovCache",
        property(lambda self: (_ for _ in ()).throw(ImportError("No module named 'CovCache'"))),
    )

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2
    assert all("cache_stats" not in res for res in results)
