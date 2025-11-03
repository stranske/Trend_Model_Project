from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import trend_analysis.multi_period.engine as engine


def test_compute_turnover_state_with_previous_allocation() -> None:
    prev_idx = np.array(["A", "B"], dtype=object)
    prev_vals = np.array([0.4, 0.6], dtype=float)
    new_series = pd.Series([0.5, 0.1], index=["A", "C"], dtype=float)

    turnover, next_idx, next_vals = engine._compute_turnover_state(
        prev_idx, prev_vals, new_series
    )

    assert pytest.approx(turnover) == 0.8
    assert next_idx.tolist() == ["A", "C"]
    assert next_vals.tolist() == [0.5, 0.1]


def test_run_schedule_with_rebalance_strategies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummySelector:
        rank_column = "score"

        def select(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            return frame, frame

    class DummyWeighting:
        def __init__(self) -> None:
            self.updates: list[tuple[pd.Series, int]] = []

        def weight(
            self, selected: pd.DataFrame, date: pd.Timestamp | None = None
        ) -> pd.DataFrame:
            del date
            weights = selected["score"].astype(float)
            weights = weights / weights.sum()
            return pd.DataFrame({"weight": weights})

        def update(self, scores: pd.Series, days: int) -> None:
            self.updates.append((scores.astype(float), days))

    def fake_apply(
        strategies: list[str],
        params: dict[str, dict[str, Any]],
        current_weights: pd.Series,
        target_weights: pd.Series,
        *,
        scores: pd.Series | None = None,
    ) -> tuple[pd.Series, float]:
        assert strategies == ["demo"]
        assert "demo" in params
        return target_weights.astype(float), 0.05

    monkeypatch.setattr(engine, "apply_rebalancing_strategies", fake_apply)
    monkeypatch.setenv("DEBUG_TURNOVER_VALIDATE", "1")

    score_frames = {
        "2020-01-31": pd.DataFrame({"score": [1.0, 2.0]}, index=["Fund A", "Fund B"]),
        "2020-02-29": pd.DataFrame({"score": [1.5, 1.0]}, index=["Fund A", "Fund C"]),
    }

    weighting = DummyWeighting()
    portfolio = engine.run_schedule(
        score_frames,
        selector=DummySelector(),
        weighting=weighting,
        rebalance_strategies=["demo"],
        rebalance_params={"demo": {"threshold": 0.1}},
    )

    assert set(portfolio.history.keys()) == {"2020-01-31", "2020-02-29"}
    assert pytest.approx(portfolio.turnover["2020-01-31"]) == 1.0
    assert portfolio.total_rebalance_costs == pytest.approx(0.1)
    assert len(weighting.updates) == 2
    # Second period should report elapsed days between the two timestamps
    assert weighting.updates[1][1] == 29


class DummyCfg:
    def __init__(self) -> None:
        self.data = {"csv_path": "unused.csv"}
        self.portfolio = {
            "policy": "",
            "selection_mode": "all",
            "random_n": 2,
            "custom_weights": None,
            "rank": None,
            "manual_list": None,
            "indices_list": None,
        }
        self.performance = {}
        self.vol_adjust = {"target_vol": 1.0}
        self.run = {"monthly_cost": 0.0}
        self.benchmarks = {}
        self.seed = 0

    def model_dump(self) -> dict[str, Any]:
        return {}


def test_run_price_frames_validation_errors() -> None:
    cfg = DummyCfg()

    with pytest.raises(TypeError):
        engine.run(cfg, price_frames=123)  # type: ignore[arg-type]

    bad_frame = pd.DataFrame({"value": [1.0]})
    with pytest.raises(ValueError):
        engine.run(cfg, price_frames={"p1": bad_frame})


def test_run_combines_price_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DummyCfg()
    cfg.performance = {"enable_cache": False}

    period = SimpleNamespace(
        in_start="2020-01",
        in_end="2020-01",
        out_start="2020-02",
        out_end="2020-02",
    )

    monkeypatch.setattr(engine, "generate_periods", lambda _: [period])

    captured: dict[str, Any] = {}

    def fake_run_analysis(
        df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        captured["df"] = df
        return {"ok": True}

    monkeypatch.setattr(engine, "_run_analysis", fake_run_analysis)

    price_frames = {
        "first": pd.DataFrame({"Date": [pd.Timestamp("2020-02-01")], "FundB": [0.2]}),
        "second": pd.DataFrame({"Date": [pd.Timestamp("2020-01-01")], "FundA": [0.1]}),
    }

    results = engine.run(cfg, price_frames=price_frames)

    expected = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")],
            "FundA": [0.1, np.nan],
            "FundB": [np.nan, 0.2],
        }
    )

    pd.testing.assert_frame_equal(captured["df"], expected, check_like=True)
    assert results[0]["period"] == (
        "2020-01",
        "2020-01",
        "2020-02",
        "2020-02",
    )


def test_run_uses_nan_policy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DummyCfg()
    cfg.data = {"csv_path": "demo.csv", "nan_policy": "drop", "nan_limit": 3}
    cfg.performance = {"enable_cache": False}

    period = SimpleNamespace(
        in_start="2020-01",
        in_end="2020-01",
        out_start="2020-02",
        out_end="2020-02",
    )
    monkeypatch.setattr(engine, "generate_periods", lambda _: [period])

    loaded = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=3, freq="ME"),
            "FundA": [0.01, 0.02, 0.03],
        }
    )

    captured: dict[str, object] = {}

    def fake_load_csv(
        path: str,
        *,
        errors: str,
        missing_policy: object,
        missing_limit: object,
    ) -> pd.DataFrame:
        captured.update(
            {
                "path": path,
                "errors": errors,
                "policy": missing_policy,
                "limit": missing_limit,
            }
        )
        return loaded

    monkeypatch.setattr(engine, "load_csv", fake_load_csv)

    policy_args: dict[str, object] = {}

    def fake_apply_missing_policy(
        frame: pd.DataFrame,
        *,
        policy: object,
        limit: object,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        policy_args["policy"] = policy
        policy_args["limit"] = limit
        return frame, {"policy": policy, "limit": limit}

    monkeypatch.setattr(engine, "apply_missing_policy", fake_apply_missing_policy)
    monkeypatch.setattr(engine, "_run_analysis", lambda *a, **k: {"ok": True})

    result = engine.run(cfg)

    assert result and captured["path"] == "demo.csv"
    assert captured["errors"] == "raise"
    assert captured["policy"] == "drop"
    assert captured["limit"] == 3
    assert policy_args == {"policy": "drop", "limit": 3}


def test_run_skips_missing_policy_with_price_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = DummyCfg()
    cfg.data = {"csv_path": "unused.csv"}
    cfg.performance = {"enable_cache": False}

    period = SimpleNamespace(
        in_start="2020-01",
        in_end="2020-01",
        out_start="2020-02",
        out_end="2020-02",
    )
    monkeypatch.setattr(engine, "generate_periods", lambda _: [period])

    called = False

    def fail_apply_missing_policy(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError(
            "apply_missing_policy should not run when price frames provided"
        )

    monkeypatch.setattr(engine, "apply_missing_policy", fail_apply_missing_policy)
    monkeypatch.setattr(
        engine, "load_csv", lambda *a, **k: (_ for _ in ()).throw(AssertionError())
    )

    captured_frames: list[pd.DataFrame] = []

    def fake_run_analysis(
        df: pd.DataFrame, *args: object, **kwargs: object
    ) -> dict[str, object]:
        captured_frames.append(df.copy())
        return {"ok": True}

    monkeypatch.setattr(engine, "_run_analysis", fake_run_analysis)

    price_frames = {
        "p1": pd.DataFrame({"Date": [pd.Timestamp("2020-01-31")], "FundA": [0.1]}),
        "p2": pd.DataFrame({"Date": [pd.Timestamp("2020-02-29")], "FundB": [0.2]}),
    }

    results = engine.run(cfg, price_frames=price_frames)

    assert results and len(captured_frames) == 1
    combined = captured_frames[0]
    assert set(combined.columns) == {"Date", "FundA", "FundB"}
    # Missing-policy helper should not run when price frames supply clean data.
    assert called is False


def test_run_incremental_covariance(monkeypatch: pytest.MonkeyPatch) -> None:
    class CovPayload:
        def __init__(self, diag: np.ndarray) -> None:
            self.cov = np.diag(diag.astype(float))

    class DummyCovCache:
        def __init__(self) -> None:
            self.incremental_updates = 0

        def stats(self) -> dict[str, int]:
            return {"updates": self.incremental_updates}

    def fake_compute(df: pd.DataFrame, *, materialise_aggregates: bool) -> CovPayload:
        diag = np.arange(1, len(df.columns) + 1, dtype=float)
        return CovPayload(diag)

    def fake_incremental(
        payload: CovPayload, old_row: np.ndarray, new_row: np.ndarray
    ) -> CovPayload:
        diag = payload.cov.diagonal() + 0.5
        return CovPayload(diag)

    import trend_analysis.perf.cache as cache_mod

    monkeypatch.setattr(cache_mod, "CovCache", DummyCovCache)
    monkeypatch.setattr(cache_mod, "compute_cov_payload", fake_compute)
    monkeypatch.setattr(cache_mod, "incremental_cov_update", fake_incremental)

    cfg = DummyCfg()
    cfg.performance = {
        "enable_cache": True,
        "incremental_cov": True,
        "shift_detection_max_steps": 3,
    }

    periods = [
        SimpleNamespace(
            in_start="2020-01",
            in_end="2020-03",
            out_start="2020-04",
            out_end="2020-05",
        ),
        SimpleNamespace(
            in_start="2020-02",
            in_end="2020-04",
            out_start="2020-05",
            out_end="2020-06",
        ),
    ]

    monkeypatch.setattr(engine, "generate_periods", lambda _: periods)

    def fake_run_analysis(
        df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(engine, "_run_analysis", fake_run_analysis)

    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Fund1": np.linspace(0.01, 0.06, num=6),
            "Fund2": np.linspace(0.03, 0.08, num=6),
        }
    )

    results = engine.run(cfg, df=df)

    assert len(results) == 2
    assert results[0]["cov_diag"] == [1.0, 2.0]
    assert results[0]["cache_stats"] == {"updates": 0}
    assert results[1]["cov_diag"] == [1.5, 2.5]
    assert results[1]["cache_stats"] == {"updates": 1}
