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


def test_run_missing_policy_removal_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DummyCfg()
    cfg.data = {"csv_path": "demo.csv", "nan_policy": "drop", "nan_limit": 2}
    cfg.performance = {}

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

    def fake_apply_missing_policy(
        frame: pd.DataFrame,
        *,
        policy: object,
        limit: object,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        empty = frame.copy()
        empty[:] = np.nan
        return empty, {"policy": policy, "limit": limit}

    monkeypatch.setattr(engine, "apply_missing_policy", fake_apply_missing_policy)

    with pytest.raises(ValueError, match="Missing-data policy removed all assets"):
        engine.run(cfg)

    assert captured == {
        "path": "demo.csv",
        "errors": "raise",
        "policy": "drop",
        "limit": 2,
    }


def test_run_threshold_hold_low_weight_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trend_analysis.core import rank_selection

    class THConfig(DummyCfg):
        def __init__(self) -> None:
            super().__init__()
            self.data = {"missing_policy": "ffill"}
            self.portfolio.update(
                {
                    "policy": "threshold_hold",
                    "threshold_hold": {
                        "metric": "Sharpe",
                        "target_n": 4,
                        "min_weight_strikes": 1,
                    },
                    "constraints": {
                        "min_weight": 0.3,
                        "max_weight": 0.6,
                        "max_funds": 4,
                        "min_weight_strikes": 1,
                    },
                    "weighting": {"name": "equal"},
                    "transaction_cost_bps": 25,
                    "max_turnover": 0.4,
                }
            )

    cfg = THConfig()

    periods = [
        SimpleNamespace(
            in_start="2020-01",
            in_end="2020-02",
            out_start="2020-03",
            out_end="2020-03",
        ),
        SimpleNamespace(
            in_start="2020-02",
            in_end="2020-03",
            out_start="2020-04",
            out_end="2020-04",
        ),
    ]

    monkeypatch.setattr(engine, "generate_periods", lambda _: periods)

    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "FundA": [0.05, 0.01, 0.08, 0.02, 0.07],
            "FundB": np.linspace(0.02, 0.06, num=5),
            "FundC": np.linspace(0.03, 0.07, num=5),
            "FundD": [0.01, 0.01, 0.01, 0.01, 0.01],
            "FundE": np.linspace(0.05, 0.09, num=5),
        }
    )

    def fake_apply_missing_policy(
        frame: pd.DataFrame,
        *,
        policy: object,
        limit: object,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        return frame, {"policy": policy, "limit": limit}

    monkeypatch.setattr(engine, "apply_missing_policy", fake_apply_missing_policy)

    def fake_metric_series(
        frame: pd.DataFrame, metric: str, stats_cfg: object
    ) -> pd.Series:
        base = np.linspace(1.0, 1.0 + frame.shape[1], num=frame.shape[1])
        return pd.Series(base, index=frame.columns, dtype=float)

    monkeypatch.setattr(rank_selection, "_compute_metric_series", fake_metric_series)

    class SelectorStub:
        def select(self, sf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            return sf, sf

    class WeightingStub:
        def __init__(self) -> None:
            self.calls = 0

        def weight(
            self, selected: pd.DataFrame, date: pd.Timestamp | None = None
        ) -> pd.DataFrame:
            del date
            self.calls += 1
            mapping = {
                1: pd.Series(
                    [0.2, 0.2, 0.2, 0.2],
                    index=["FundA", "FundB", "FundC", "FundD"],
                ),
                2: pd.Series(
                    [0.05, 0.55, 0.2, 0.2],
                    index=["FundA", "FundB", "FundC", "FundD"],
                ),
                3: pd.Series(
                    [0.4, 0.3, 0.2, 0.1],
                    index=["FundB", "FundC", "FundD", "FundE"],
                ),
            }
            choice = mapping.get(self.calls, mapping[max(mapping)])
            weights = choice.reindex(selected.index).fillna(0.1)
            return pd.DataFrame({"weight": weights.astype(float)})

    weighting_stub = WeightingStub()
    monkeypatch.setattr(engine, "EqualWeight", lambda: weighting_stub)
    selector_stub = SelectorStub()
    from trend_analysis import selector as selector_mod

    monkeypatch.setattr(
        selector_mod, "create_selector_by_name", lambda *a, **k: selector_stub
    )

    class RebalancerStub:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.calls = 0

        def apply_triggers(
            self, prev_weights: pd.Series, score_frame: pd.DataFrame
        ) -> pd.Series:
            del score_frame
            self.calls += 1
            return prev_weights

    reb_stub = RebalancerStub()
    monkeypatch.setattr(engine, "Rebalancer", lambda *a, **k: reb_stub)

    analysis_calls: list[list[str] | None] = []

    def fake_run_analysis(
        df_arg: pd.DataFrame,
        in_start: str,
        in_end: str,
        out_start: str,
        out_end: str,
        *_args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del df_arg, in_start, in_end, out_start, out_end
        analysis_calls.append(kwargs.get("manual_funds"))
        return {"ok": True}

    monkeypatch.setattr(engine, "_run_analysis", fake_run_analysis)

    results = engine.run(cfg, df=df)

    assert len(results) == 2
    assert "FundA" in analysis_calls[0]
    assert "FundE" in analysis_calls[1]

    events = results[1]["manager_changes"]
    drop_event = next(
        e for e in events if e["manager"] == "FundA" and e["action"] == "dropped"
    )
    assert drop_event["reason"] == "low_weight_strikes"
    add_event = next(
        e for e in events if e["manager"] == "FundE" and e["action"] == "added"
    )
    assert add_event["reason"] == "replacement"

    turnover_cap = cfg.portfolio["max_turnover"]
    assert results[1]["turnover"] <= turnover_cap + 1e-9
    expected_cost = results[1]["turnover"] * (
        cfg.portfolio["transaction_cost_bps"] / 10000
    )
    assert results[1]["transaction_cost"] == pytest.approx(expected_cost)


def test_run_threshold_hold_weight_bounds_fill_deficit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trend_analysis.core import rank_selection

    class THConfig(DummyCfg):
        def __init__(self) -> None:
            super().__init__()
            self.data = {"missing_policy": "ffill"}
            self.portfolio.update(
                {
                    "policy": "threshold_hold",
                    "threshold_hold": {"metric": "Sharpe", "target_n": 2},
                    "constraints": {
                        "min_weight": 0.3,
                        "max_weight": 0.6,
                        "max_funds": 2,
                    },
                    "weighting": {"name": "equal"},
                    "max_turnover": 1.0,
                }
            )

    cfg = THConfig()

    period = SimpleNamespace(
        in_start="2020-01",
        in_end="2020-02",
        out_start="2020-03",
        out_end="2020-03",
    )

    monkeypatch.setattr(engine, "generate_periods", lambda _: [period])

    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=3, freq="ME"),
            "FundA": [0.01, 0.02, 0.015],
            "FundB": [0.03, 0.025, 0.02],
            "FundC": [0.04, 0.045, 0.05],
        }
    )

    monkeypatch.setattr(
        engine,
        "apply_missing_policy",
        lambda frame, *, policy, limit: (frame, {"policy": policy, "limit": limit}),
    )

    def fake_metric_series(
        frame: pd.DataFrame, metric: str, stats_cfg: object
    ) -> pd.Series:
        base = np.linspace(1.0, 1.0 + frame.shape[1], num=frame.shape[1])
        return pd.Series(base, index=frame.columns, dtype=float)

    monkeypatch.setattr(rank_selection, "_compute_metric_series", fake_metric_series)

    class SelectorStub:
        def select(self, sf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            return sf, sf

    selector_stub = SelectorStub()
    from trend_analysis import selector as selector_mod

    monkeypatch.setattr(
        selector_mod, "create_selector_by_name", lambda *a, **k: selector_stub
    )

    class WeightingStub:
        def __init__(self) -> None:
            self.calls = 0

        def weight(
            self, selected: pd.DataFrame, date: pd.Timestamp | None = None
        ) -> pd.DataFrame:
            del date
            self.calls += 1
            base = pd.Series(0.1, index=selected.index, dtype=float)
            return pd.DataFrame({"weight": base})

    monkeypatch.setattr(engine, "EqualWeight", lambda: WeightingStub())
    monkeypatch.setattr(
        engine,
        "Rebalancer",
        lambda *a, **k: SimpleNamespace(apply_triggers=lambda w, sf: w),
    )

    captured_weights: list[dict[str, float] | None] = []

    def fake_run_analysis(
        df_arg: pd.DataFrame,
        *_args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del df_arg
        captured_weights.append(kwargs.get("custom_weights"))
        return {"ok": True}

    monkeypatch.setattr(engine, "_run_analysis", fake_run_analysis)

    results = engine.run(cfg, df=df)

    assert len(results) == 1
    final_weights = captured_weights[0]
    assert final_weights is not None
    # custom weights expressed in percentages; ensure they sum to 100%
    assert pytest.approx(sum(final_weights.values()), rel=1e-9) == 100.0


def test_run_threshold_hold_reseeds_and_skips_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trend_analysis.core import rank_selection

    class THConfig(DummyCfg):
        def __init__(self) -> None:
            super().__init__()
            self.data = {"missing_policy": "ffill"}
            self.portfolio.update(
                {
                    "policy": "threshold_hold",
                    "threshold_hold": {"metric": "Sharpe", "target_n": 3},
                    "constraints": {
                        "min_weight": 0.2,
                        "max_weight": 0.6,
                        "max_funds": 3,
                    },
                    "weighting": {"name": "equal"},
                    "max_turnover": 1.0,
                }
            )

    cfg = THConfig()

    periods = [
        SimpleNamespace(
            in_start="2020-01",
            in_end="2020-02",
            out_start="2020-03",
            out_end="2020-03",
        ),
        SimpleNamespace(
            in_start="2020-02",
            in_end="2020-03",
            out_start="2020-04",
            out_end="2020-04",
        ),
    ]

    monkeypatch.setattr(engine, "generate_periods", lambda _: periods)

    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=4, freq="ME"),
            "FundA": [0.03, 0.04, 0.05, 0.06],
            "FundB": [0.025, 0.03, 0.028, 0.029],
            "FundC": [0.035, 0.04, 0.038, 0.039],
            "FundD": [0.01, 0.01, 0.01, 0.01],
        }
    )

    monkeypatch.setattr(
        engine,
        "apply_missing_policy",
        lambda frame, *, policy, limit: (frame, {"policy": policy, "limit": limit}),
    )

    def fake_metric_series(
        frame: pd.DataFrame, metric: str, stats_cfg: object
    ) -> pd.Series:
        base = np.linspace(1.0, 1.0 + frame.shape[1], num=frame.shape[1])
        return pd.Series(base, index=frame.columns, dtype=float)

    monkeypatch.setattr(rank_selection, "_compute_metric_series", fake_metric_series)

    class SelectorStub:
        def select(self, sf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            return sf, sf

    selector_stub = SelectorStub()
    from trend_analysis import selector as selector_mod

    monkeypatch.setattr(
        selector_mod, "create_selector_by_name", lambda *a, **k: selector_stub
    )

    class WeightingStub:
        def __init__(self) -> None:
            self.calls = 0

        def weight(
            self, selected: pd.DataFrame, date: pd.Timestamp | None = None
        ) -> pd.DataFrame:
            del date
            self.calls += 1
            mapping = {
                1: pd.Series([0.4, 0.35, 0.25], index=["FundA", "FundB", "FundC"]),
                2: pd.Series([0.6, 0.4], index=["FundB", "FundC"]),
            }
            choice = mapping.get(self.calls, mapping[max(mapping)])
            weights = choice.reindex(selected.index).fillna(0.2)
            return pd.DataFrame({"weight": weights.astype(float)})

    weighting_stub = WeightingStub()
    monkeypatch.setattr(engine, "EqualWeight", lambda: weighting_stub)

    class RebalancerStub:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.calls = 0

        def apply_triggers(
            self, prev_weights: pd.Series, score_frame: pd.DataFrame
        ) -> pd.Series:
            del score_frame
            self.calls += 1
            if self.calls == 1:
                return prev_weights
            return pd.Series([0.2], index=["Ghost"], dtype=float)

    reb_stub = RebalancerStub()
    monkeypatch.setattr(engine, "Rebalancer", lambda *a, **k: reb_stub)

    analysis_calls: list[list[str] | None] = []

    def fake_run_analysis(
        df_arg: pd.DataFrame,
        *_args: Any,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        del df_arg
        analysis_calls.append(kwargs.get("manual_funds"))
        if len(analysis_calls) == 2:
            return None
        return {"ok": True}

    monkeypatch.setattr(engine, "_run_analysis", fake_run_analysis)

    results = engine.run(cfg, df=df)

    assert len(results) == 1
    assert set(analysis_calls[0]) == {"FundA", "FundB", "FundC"}
    reseat_events = results[0]["manager_changes"]
    assert any(e["reason"] == "seed" for e in reseat_events)
    # Second period is skipped, but reseed should have been attempted
    assert "FundB" in analysis_calls[1]
    assert "FundC" in analysis_calls[1]
