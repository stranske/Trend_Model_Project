from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd
import pytest

from trend_analysis.multi_period import engine as mp_engine


@dataclass
class DummyConfig:
    """Minimal configuration object for exercising threshold-hold branches."""

    multi_period: Dict[str, Any] = field(
        default_factory=lambda: {
            "frequency": "M",
            "in_sample_len": 2,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-05",
        }
    )
    data: Dict[str, Any] = field(
        default_factory=lambda: {
            "csv_path": "unused.csv",
            "allow_risk_free_fallback": True,
        }
    )
    portfolio: Dict[str, Any] = field(
        default_factory=lambda: {
            "policy": "threshold_hold",
            "random_n": 4,
            "transaction_cost_bps": 0.0,
            "max_turnover": 1.0,
            "threshold_hold": {
                "target_n": 3,
                "metric": "Sharpe",
                "z_exit_soft": -5.0,
                "z_entry_soft": -5.0,
            },
            "constraints": {
                "max_funds": 3,
                "min_weight": 0.05,
                "max_weight": 0.8,
                "min_weight_strikes": 2,
            },
            "weighting": {"name": "adaptive_bayes", "params": {}},
            "indices_list": None,
        }
    )
    vol_adjust: Dict[str, Any] = field(default_factory=lambda: {"target_vol": 1.0})
    benchmarks: Dict[str, Any] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=lambda: {"monthly_cost": 0.0})
    seed: int = 123

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


class SequenceWeighting:
    """Deterministic weighting helper providing scripted outputs."""

    def __init__(self, sequences: Sequence[Dict[str, float]]) -> None:
        self._sequences = list(sequences) or [{}]
        self.calls = 0
        self.update_calls: list[tuple[pd.Series, int]] = []

    def weight(
        self, selected: pd.DataFrame, date: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        del date
        seq = self._sequences[min(self.calls, len(self._sequences) - 1)]
        weights = pd.Series(
            {idx: seq.get(idx, 0.05) for idx in selected.index},
            index=selected.index,
            dtype=float,
        )
        self.calls += 1
        return weights.to_frame("weight")

    def update(self, scores: pd.Series, days: int) -> None:
        self.update_calls.append((scores.astype(float), int(days)))


class ScriptedSelector:
    """Selector that preserves a provided fund ordering."""

    rank_column = "Sharpe"

    def __init__(self, ordering: Iterable[str]) -> None:
        self._ordering = list(ordering)

    def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        ordered = [ix for ix in self._ordering if ix in score_frame.index]
        selected = score_frame.loc[ordered]
        return selected, selected


class IdentityRebalancer:
    def __init__(self, *_cfg: Any) -> None:
        self.calls: list[pd.Series] = []

    def apply_triggers(
        self, prev_weights: pd.Series, _sf: pd.DataFrame, **kwargs
    ) -> pd.Series:
        self.calls.append(prev_weights.copy())
        return prev_weights.astype(float)


def _stub_run_analysis(call_log: list[Dict[str, Any]]):
    def _run(
        *_args: Any,
        manual_funds: Sequence[str] | None = None,
        custom_weights: Dict[str, float] | None = None,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        call_log.append(
            {
                "manual_funds": list(manual_funds or []),
                "custom_weights": dict(custom_weights or {}),
            }
        )
        return {
            "selected_funds": list(manual_funds or []),
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

    return _run


def test_threshold_hold_yields_placeholder_when_universe_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = DummyConfig()
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "Alpha One": [0.05, 0.06, float("nan")],
            "Beta One": [float("nan"), 0.07, 0.08],
        }
    )

    periods = [DummyPeriod("2020-01-31", "2020-02-29", "2020-03-31", "2020-03-31")]

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    run_calls: list[Dict[str, Any]] = []
    monkeypatch.setattr(mp_engine, "_run_analysis", _stub_run_analysis(run_calls))

    results = mp_engine.run(cfg, df=df)

    assert run_calls == []
    assert len(results) == 1
    result = results[0]
    assert result["selected_funds"] == []
    assert isinstance(result["score_frame"], pd.DataFrame)
    assert result["score_frame"].empty
    assert result["manager_changes"] == []
    assert result["out_ew_stats"] is None
    assert result["out_user_stats"] is None


def test_threshold_hold_drops_low_weight_and_replenishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = DummyConfig()
    cfg.portfolio["threshold_hold"].update({"target_n": 3, "metric": "Sharpe"})
    cfg.portfolio["constraints"].update(
        {"max_funds": 3, "min_weight": 0.05, "max_weight": 0.8, "min_weight_strikes": 2}
    )

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
            "Alpha One": [0.04, 0.03, 0.02, 0.01, 0.03],
            "Alpha Two": [0.05, 0.04, 0.03, 0.02, 0.02],
            "Beta One": [0.02, 0.03, 0.02, 0.01, 0.02],
            "Gamma One": [0.03, 0.03, 0.02, 0.02, 0.01],
            "Delta One": [0.01, 0.02, 0.03, 0.04, 0.05],
        }
    )

    periods = [
        DummyPeriod("2020-01-31", "2020-03-31", "2020-04-30", "2020-04-30"),
        DummyPeriod("2020-02-29", "2020-04-30", "2020-05-31", "2020-05-31"),
    ]
    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    metric_maps = {
        "AnnualReturn": {
            "Alpha One": 0.11,
            "Alpha Two": 0.12,
            "Beta One": 0.07,
            "Gamma One": 0.1,
            "Delta One": 0.05,
        },
        "Volatility": {
            "Alpha One": 0.2,
            "Alpha Two": 0.18,
            "Beta One": 0.15,
            "Gamma One": 0.16,
            "Delta One": 0.22,
        },
        "Sharpe": {
            "Alpha One": 1.1,
            "Alpha Two": 1.2,
            "Beta One": 0.5,
            "Gamma One": 1.15,
            "Delta One": 0.4,
        },
        "Sortino": {
            "Alpha One": 1.3,
            "Alpha Two": 1.4,
            "Beta One": 0.6,
            "Gamma One": 1.25,
            "Delta One": 0.45,
        },
        "InformationRatio": {
            "Alpha One": 0.9,
            "Alpha Two": 1.0,
            "Beta One": 0.4,
            "Gamma One": 0.85,
            "Delta One": 0.35,
        },
        "MaxDrawdown": {
            "Alpha One": -0.12,
            "Alpha Two": -0.11,
            "Beta One": -0.05,
            "Gamma One": -0.08,
            "Delta One": -0.09,
        },
    }

    import trend_analysis.core.rank_selection as rank_sel

    def fake_metric_series(frame: pd.DataFrame, metric: str, _cfg: Any) -> pd.Series:
        values = metric_maps[metric]
        return pd.Series({col: values[col] for col in frame.columns}, dtype=float)

    monkeypatch.setattr(rank_sel, "_compute_metric_series", fake_metric_series)

    weighting = SequenceWeighting(
        [
            {
                "Alpha One": 0.55,
                "Alpha Two": 0.25,
                "Beta One": 0.15,
                "Gamma One": 0.04,
                "Delta One": 0.01,
            },
            {"Alpha Two": 0.6, "Gamma One": 0.02, "Beta One": 0.38},
            {"Alpha Two": 0.55, "Gamma One": 0.02, "Beta One": 0.4},
            {"Alpha Two": 0.52, "Beta One": 0.28, "Delta One": 0.2},
        ]
    )
    monkeypatch.setattr(mp_engine, "AdaptiveBayesWeighting", lambda *a, **k: weighting)
    monkeypatch.setattr(mp_engine, "Rebalancer", IdentityRebalancer)

    import trend_analysis.selector as selector_mod

    selector = ScriptedSelector(
        ["Alpha One", "Alpha Two", "Beta One", "Gamma One", "Delta One"]
    )
    monkeypatch.setattr(
        selector_mod, "create_selector_by_name", lambda *a, **k: selector
    )

    run_calls: list[Dict[str, Any]] = []
    monkeypatch.setattr(mp_engine, "_run_analysis", _stub_run_analysis(run_calls))

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2
    manual_funds = run_calls[1]["manual_funds"]
    assert len(manual_funds) == 3
    assert "Alpha Two" in manual_funds
    assert "Delta One" in manual_funds
    assert "Gamma One" in manual_funds

    changes = results[1]["manager_changes"]
    reasons = {change["reason"] for change in changes}
    assert "low_weight_strikes" in reasons
    assert any(
        change["manager"] == "Gamma One" and change["reason"] == "low_weight_strikes"
        for change in changes
    )
    assert any(
        change["manager"] == "Gamma One" and change["reason"] == "replacement"
        for change in changes
    )


def test_threshold_hold_scales_trades_to_respect_turnover_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = DummyConfig()
    cfg.portfolio["threshold_hold"].update({"target_n": 2, "metric": "Sharpe"})
    cfg.portfolio["constraints"].update(
        {"max_funds": 2, "min_weight": 0.0, "max_weight": 1.0, "min_weight_strikes": 2}
    )
    cfg.portfolio.update({"transaction_cost_bps": 15.0, "max_turnover": 0.4})

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
            "Alpha One": [0.05, 0.04, 0.03, 0.02],
            "Beta One": [0.01, 0.02, 0.03, 0.04],
            "Cash Proxy": [0.0, 0.0, 0.0, 0.0],
        }
    )

    periods = [
        DummyPeriod("2020-01-31", "2020-02-29", "2020-03-31", "2020-03-31"),
        DummyPeriod("2020-02-29", "2020-03-31", "2020-04-30", "2020-04-30"),
    ]
    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    metric_maps = {
        "AnnualReturn": {"Alpha One": 0.1, "Beta One": 0.08, "Cash Proxy": 0.01},
        "Volatility": {"Alpha One": 0.2, "Beta One": 0.18, "Cash Proxy": 0.001},
        "Sharpe": {"Alpha One": 1.0, "Beta One": 0.9, "Cash Proxy": 0.05},
        "Sortino": {"Alpha One": 1.1, "Beta One": 1.0, "Cash Proxy": 0.05},
        "InformationRatio": {"Alpha One": 0.6, "Beta One": 0.55, "Cash Proxy": 0.02},
        "MaxDrawdown": {"Alpha One": -0.1, "Beta One": -0.08, "Cash Proxy": -0.01},
    }

    import trend_analysis.core.rank_selection as rank_sel

    def fake_metric_series(frame: pd.DataFrame, metric: str, _cfg: Any) -> pd.Series:
        values = metric_maps[metric]
        return pd.Series({col: values[col] for col in frame.columns}, dtype=float)

    monkeypatch.setattr(rank_sel, "_compute_metric_series", fake_metric_series)

    weighting = SequenceWeighting(
        [
            {"Alpha One": 0.6, "Beta One": 0.4},
            {"Alpha One": 0.6, "Beta One": 0.4},
            {"Alpha One": 0.0, "Beta One": 1.0},
        ]
    )
    monkeypatch.setattr(mp_engine, "AdaptiveBayesWeighting", lambda *a, **k: weighting)
    monkeypatch.setattr(mp_engine, "Rebalancer", IdentityRebalancer)

    import trend_analysis.selector as selector_mod

    selector = ScriptedSelector(["Alpha One", "Beta One"])
    monkeypatch.setattr(
        selector_mod, "create_selector_by_name", lambda *a, **k: selector
    )

    run_calls: list[Dict[str, Any]] = []
    monkeypatch.setattr(mp_engine, "_run_analysis", _stub_run_analysis(run_calls))

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2
    # The expected weights below are derived from the initial weights and the turnover cap.
    # Initial weights: {"Alpha One": 0.6, "Beta One": 0.4}
    # Turnover cap: 1.0 (from config), but the test scenario results in a turnover of 0.4.
    # The weights are scaled such that Alpha One: 0.325, Beta One: 0.675, then multiplied by 100 for percentage.
    expected_weights = {
        "Alpha One": pytest.approx(0.325 * 100, rel=1e-3),
        "Beta One": pytest.approx(0.675 * 100, rel=1e-3),
    }
    assert run_calls[1]["custom_weights"] == expected_weights


def test_threshold_hold_seed_dedupe_and_rebalance_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = DummyConfig()
    cfg.portfolio["threshold_hold"].update(
        {
            "target_n": 4,
            "metric": "Sharpe",
            "z_exit_soft": -0.25,
            "z_entry_soft": 0.5,
        }
    )
    cfg.portfolio["constraints"].update(
        {"max_funds": 2, "min_weight": 0.0, "max_weight": 1.0, "min_weight_strikes": 2}
    )
    cfg.portfolio["weighting"] = {"name": "equal", "params": {}}

    periods = [
        DummyPeriod("2020-01-31", "2020-02-29", "2020-03-31", "2020-03-31"),
        DummyPeriod("2020-02-29", "2020-03-31", "2020-04-30", "2020-04-30"),
    ]
    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"])
    df = pd.DataFrame(
        {
            "Date": dates,
            "Alpha One": [0.04, 0.03, 0.02, 0.01],
            "Alpha Two": [0.05, 0.04, 0.03, 0.02],
            "Beta One": [0.02, 0.03, 0.01, 0.0],
            "Gamma One": [0.01, 0.02, 0.015, 0.02],
            "Delta One": [0.005, 0.006, 0.007, 0.008],
            "RF Proxy": [0.0, 0.0, 0.0, 0.0],
        }
    )

    metric_queue = [
        {
            "AnnualReturn": {
                "Alpha One": 0.12,
                "Alpha Two": 0.09,
                "Beta One": 0.11,
                "Gamma One": 0.07,
                "Delta One": 0.05,
            },
            "Volatility": {
                "Alpha One": 0.18,
                "Alpha Two": 0.2,
                "Beta One": 0.16,
                "Gamma One": 0.19,
                "Delta One": 0.22,
            },
            "Sharpe": {
                "Alpha One": 1.5,
                "Alpha Two": 1.0,
                "Beta One": 1.3,
                "Gamma One": 0.8,
                "Delta One": 0.6,
            },
            "Sortino": {
                "Alpha One": 1.6,
                "Alpha Two": 1.1,
                "Beta One": 1.35,
                "Gamma One": 0.82,
                "Delta One": 0.65,
            },
            "InformationRatio": {
                "Alpha One": 0.9,
                "Alpha Two": 0.6,
                "Beta One": 0.75,
                "Gamma One": 0.4,
                "Delta One": 0.35,
            },
            "MaxDrawdown": {
                "Alpha One": -0.08,
                "Alpha Two": -0.1,
                "Beta One": -0.09,
                "Gamma One": -0.12,
                "Delta One": -0.13,
            },
        },
        {
            "AnnualReturn": {
                "Alpha One": 0.1,
                "Alpha Two": 0.08,
                "Beta One": -0.02,
                "Gamma One": 0.06,
                "Delta One": 0.14,
            },
            "Volatility": {
                "Alpha One": 0.17,
                "Alpha Two": 0.19,
                "Beta One": 0.25,
                "Gamma One": 0.2,
                "Delta One": 0.18,
            },
            "Sharpe": {
                "Alpha One": 1.1,
                "Alpha Two": 0.7,
                "Beta One": -0.5,
                "Gamma One": 0.9,
                "Delta One": 1.8,
            },
            "Sortino": {
                "Alpha One": 1.2,
                "Alpha Two": 0.75,
                "Beta One": -0.45,
                "Gamma One": 0.95,
                "Delta One": 1.85,
            },
            "InformationRatio": {
                "Alpha One": 0.8,
                "Alpha Two": 0.5,
                "Beta One": -0.3,
                "Gamma One": 0.6,
                "Delta One": 1.0,
            },
            "MaxDrawdown": {
                "Alpha One": -0.07,
                "Alpha Two": -0.09,
                "Beta One": -0.2,
                "Gamma One": -0.1,
                "Delta One": -0.05,
            },
        },
    ]
    call_state = {"count": 0}

    import trend_analysis.core.rank_selection as rank_sel

    def fake_metric_series(frame: pd.DataFrame, metric: str, _cfg: Any) -> pd.Series:
        period_idx = call_state["count"] // 6
        values = metric_queue[period_idx][metric]
        call_state["count"] += 1
        return pd.Series({col: values[col] for col in frame.columns}, dtype=float)

    monkeypatch.setattr(rank_sel, "_compute_metric_series", fake_metric_series)

    import trend_analysis.selector as selector_mod

    class OrderedSelector:
        rank_column = "Sharpe"

        def __init__(self, ordering: Sequence[str]) -> None:
            self._ordering = list(ordering)

        def select(
            self, score_frame: pd.DataFrame
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            ordered = [fund for fund in self._ordering if fund in score_frame.index]
            selected = score_frame.loc[ordered]
            return selected, selected

    monkeypatch.setattr(
        selector_mod,
        "create_selector_by_name",
        lambda *args, **kwargs: OrderedSelector(
            ["Alpha Two", "Alpha One", "Beta One", "Gamma One", "Delta One"]
        ),
    )

    class ScriptedRebalancer:
        def __init__(self, *_cfg: Any) -> None:
            self.calls = 0

        def apply_triggers(
            self, prev_weights: pd.Series, _sf: pd.DataFrame, **kwargs: Any
        ) -> pd.Series:
            self.calls += 1
            series = prev_weights.astype(float).copy()
            if self.calls == 1:
                if "Alpha One" in series.index:
                    series.loc["Alpha One"] = 0.6
                if "Beta One" in series.index:
                    series = series.drop("Beta One")
                series.loc["Delta One"] = 0.4
            return series

    monkeypatch.setattr(mp_engine, "Rebalancer", ScriptedRebalancer)

    run_calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(mp_engine, "_run_analysis", _stub_run_analysis(run_calls))

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2
    first_manual = run_calls[0]["manual_funds"]
    assert first_manual == ["Alpha One", "Beta One"]
    assert "Alpha Two" not in first_manual
    assert "Delta One" not in first_manual

    second_manual = run_calls[1]["manual_funds"]
    assert len(second_manual) == 2
    assert set(second_manual) == {"Alpha One", "Delta One"}

    events = results[1]["manager_changes"]
    dropped = next(
        change
        for change in events
        if change["manager"] == "Beta One" and change["action"] == "dropped"
    )
    added = next(
        change
        for change in events
        if change["manager"] == "Delta One" and change["action"] == "added"
    )
    assert dropped["reason"] == "z_exit"
    assert added["reason"] == "z_entry"


def test_run_schedule_applies_strategy_and_turnover_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score_frames = {
        "2020-01-31": pd.DataFrame(
            {"Sharpe": [1.0, 0.5]}, index=["Alpha One", "Beta One"]
        ),
        "2020-02-29": pd.DataFrame(
            {"Sharpe": [0.3, 1.2]}, index=["Alpha One", "Beta One"]
        ),
    }

    selector = ScriptedSelector(["Alpha One", "Beta One"])
    weighting = SequenceWeighting(
        [{"Alpha One": 0.55, "Beta One": 0.45}, {"Alpha One": 0.2, "Beta One": 0.8}]
    )

    apply_calls: list[Dict[str, Any]] = []

    def fake_apply(
        strategies: List[str],
        params: Dict[str, Dict[str, Any]],
        current: pd.Series,
        target: pd.Series,
        *,
        scores: pd.Series | None = None,
    ) -> tuple[pd.Series, float]:
        assert strategies == ["one"]
        assert "one" in params
        if not apply_calls:
            final = pd.Series({"Alpha One": 0.6, "Beta One": 0.4}, dtype=float)
            cost = 1.25
        else:
            final = pd.Series({"Alpha One": 0.25, "Beta One": 0.75}, dtype=float)
            cost = 2.5
        apply_calls.append(
            {
                "current": current.astype(float),
                "target": target.astype(float),
                "scores": None if scores is None else scores.astype(float),
            }
        )
        return final, cost

    monkeypatch.setattr(mp_engine, "apply_rebalancing_strategies", fake_apply)

    portfolio = mp_engine.run_schedule(
        score_frames,
        selector,
        weighting,
        rank_column="Sharpe",
        rebalancer=None,
        rebalance_strategies=["one"],
        rebalance_params={"one": {"alpha": 1.0}},
    )

    assert len(apply_calls) == 2
    assert apply_calls[0]["current"].empty
    pd.testing.assert_series_equal(
        apply_calls[1]["current"],
        pd.Series({"Alpha One": 0.6, "Beta One": 0.4}, dtype=float),
    )
    for call in apply_calls:
        assert list(call["target"].index) == ["Alpha One", "Beta One"]
        assert isinstance(call["scores"], pd.Series)

    assert portfolio.turnover["2020-01-31"] == pytest.approx(1.0)
    assert portfolio.turnover["2020-02-29"] == pytest.approx(0.7, rel=1e-9)
    assert portfolio.costs["2020-01-31"] == pytest.approx(1.25)
    assert portfolio.costs["2020-02-29"] == pytest.approx(2.5)

    assert len(weighting.update_calls) == 2
    assert weighting.update_calls[0][1] == 0
    assert weighting.update_calls[1][1] > 0


def test_threshold_hold_enforces_bounds_and_replacement_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise weight-bound enforcement, dedupe, and replacement logic."""

    cfg = DummyConfig()
    cfg.portfolio["threshold_hold"].update(
        {
            "target_n": 5,
            "metric": "Sharpe",
            "z_exit_soft": -0.5,
            "z_entry_soft": -0.5,
        }
    )
    cfg.portfolio["constraints"].update(
        {
            "max_funds": 3,
            "min_weight": 0.05,
            "max_weight": 0.6,
            "min_weight_strikes": 2,
        }
    )
    cfg.portfolio.update(
        {
            "transaction_cost_bps": 25.0,
            "max_turnover": 0.2,
            "indices_list": ["Index Bench"],
        }
    )

    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    df = pd.DataFrame(
        {
            "Date": [d.strftime("%Y-%m-%d") for d in dates],
            "Alpha One": [0.05, 0.04, 0.03, 0.02, 0.01, 0.02],
            "Alpha Two": [0.03, 0.02, 0.01, 0.02, 0.01, 0.02],
            "Bravo One": [0.04, 0.03, 0.05, 0.04, 0.03, 0.02],
            "Charlie One": [0.01, 0.02, 0.01, 0.02, 0.01, 0.02],
            "Delta One": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Index Bench": [0.0] * 6,
        }
    )

    periods = [
        DummyPeriod("2020-01-31", "2020-03-31", "2020-04-30", "2020-04-30"),
        DummyPeriod("2020-02-29", "2020-04-30", "2020-05-31", "2020-05-31"),
    ]

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    metrics = {
        "AnnualReturn": {
            "Alpha One": 0.18,
            "Alpha Two": 0.12,
            "Bravo One": 0.16,
            "Charlie One": 0.11,
            "Delta One": 0.10,
        },
        "Volatility": {
            "Alpha One": 0.20,
            "Alpha Two": 0.18,
            "Bravo One": 0.17,
            "Charlie One": 0.15,
            "Delta One": 0.16,
        },
        "Sharpe": {
            "Alpha One": 1.60,
            "Alpha Two": 1.30,
            "Bravo One": 1.50,
            "Charlie One": 1.35,
            "Delta One": 1.05,
        },
        "Sortino": {
            "Alpha One": 1.70,
            "Alpha Two": 1.35,
            "Bravo One": 1.55,
            "Charlie One": 1.40,
            "Delta One": 1.05,
        },
        "InformationRatio": {
            "Alpha One": 1.40,
            "Alpha Two": 1.10,
            "Bravo One": 1.35,
            "Charlie One": 1.15,
            "Delta One": 0.90,
        },
        "MaxDrawdown": {
            "Alpha One": -0.10,
            "Alpha Two": -0.08,
            "Bravo One": -0.09,
            "Charlie One": -0.07,
            "Delta One": -0.06,
        },
    }

    from trend_analysis import selector as selector_mod
    from trend_analysis.core import rank_selection as rank_mod

    def fake_metric_series(frame: pd.DataFrame, metric: str, _cfg: Any) -> pd.Series:
        values = metrics[metric]
        return pd.Series({col: values[col] for col in frame.columns}, dtype=float)

    monkeypatch.setattr(rank_mod, "_compute_metric_series", fake_metric_series)

    class OrderedSelector:
        def __init__(self, top_n: int, rank_column: str) -> None:
            self.top_n = top_n
            self.rank_column = rank_column

        def select(
            self, score_frame: pd.DataFrame
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            ordered = score_frame.sort_values(self.rank_column, ascending=False)
            picked = ordered.head(self.top_n)
            return picked, picked

    monkeypatch.setattr(
        selector_mod,
        "create_selector_by_name",
        lambda _name, *, top_n, rank_column: OrderedSelector(top_n, rank_column),
    )

    class ScriptedWeighting:
        def __init__(self, *_, **__) -> None:
            self.calls = 0

        def weight(
            self, selected: pd.DataFrame, date: pd.Timestamp | None = None
        ) -> pd.DataFrame:
            del date
            sequences = [
                {
                    "Alpha One": 0.90,
                    "Alpha Two": 0.40,
                    "Bravo One": 0.90,
                    "Charlie One": 0.01,
                    "Delta One": 0.20,
                },
                {"Alpha One": 0.90, "Bravo One": 0.90, "Charlie One": 0.01},
                {"Alpha One": 0.70, "Charlie One": 0.02},
                {"Alpha One": 0.60, "Bravo One": 0.25, "Charlie One": 0.15},
            ]
            data = sequences[min(self.calls, len(sequences) - 1)]
            weights = pd.Series(
                [data.get(idx, 0.05) for idx in selected.index],
                index=selected.index,
                dtype=float,
            )
            self.calls += 1
            return weights.to_frame("weight")

        def update(self, scores: pd.Series, days: int) -> None:
            return None

    monkeypatch.setattr(mp_engine, "AdaptiveBayesWeighting", ScriptedWeighting)

    class ScriptedRebalancer:
        def __init__(self, *_cfg: Any) -> None:
            self.invocations: list[pd.Series] = []

        def apply_triggers(
            self, prev_weights: pd.Series, score_frame: pd.DataFrame, **kwargs: Any
        ) -> pd.Series:
            self.invocations.append(prev_weights.copy())
            return pd.Series(
                {
                    "Alpha One": float(prev_weights.get("Alpha One", 0.45)),
                    "Alpha Two": 0.20,
                    "Charlie One": float(prev_weights.get("Charlie One", 0.02)),
                }
            )

    monkeypatch.setattr(mp_engine, "Rebalancer", ScriptedRebalancer)

    run_calls: list[Dict[str, Any]] = []

    def fake_run_analysis(
        *_args: Any,
        manual_funds: Sequence[str] | None = None,
        custom_weights: Dict[str, float] | None = None,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        run_calls.append(
            {
                "manual_funds": list(manual_funds or []),
                "custom_weights": dict(custom_weights or {}),
            }
        )
        return {
            "selected_funds": list(manual_funds or []),
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

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2
    assert [len(call["manual_funds"]) for call in run_calls] == [3, 2]
    second_weights = run_calls[1]["custom_weights"]
    assert set(second_weights) == {"Alpha One", "Bravo One", "Charlie One"}
    assert any(event["reason"] == "seed" for event in results[0]["manager_changes"])

    second_events = results[1]["manager_changes"]
    reasons = {event["reason"] for event in second_events}
    assert {"one_per_firm", "low_weight_strikes", "replacement"} <= reasons
    assert results[1]["turnover"] > 0


def test_threshold_hold_weight_bounds_handles_uniform_minimum(monkeypatch):
    """All holdings floored at the minimum weight trigger the excess branch."""

    cfg = DummyConfig()
    cfg.portfolio["threshold_hold"].update({"target_n": 3, "metric": "Sharpe"})
    cfg.portfolio["constraints"].update(
        {
            "max_funds": 3,
            "min_weight": 0.4,
            "max_weight": 0.6,
            "min_weight_strikes": 10,
        }
    )

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
            "Alpha One": [0.05, 0.04, 0.03, 0.02],
            "Bravo One": [0.02, 0.03, 0.01, 0.02],
            "Charlie One": [0.01, 0.02, 0.01, 0.02],
            "Cash": [0.0, 0.0, 0.0, 0.0],
        }
    )

    periods = [
        DummyPeriod("2020-01-31", "2020-03-31", "2020-04-30", "2020-04-30"),
    ]

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    metrics = {
        "AnnualReturn": {
            "Alpha One": 0.18,
            "Bravo One": 0.16,
            "Charlie One": 0.15,
        },
        "Volatility": {
            "Alpha One": 0.20,
            "Bravo One": 0.22,
            "Charlie One": 0.21,
        },
        "Sharpe": {
            "Alpha One": 1.5,
            "Bravo One": 1.1,
            "Charlie One": 1.0,
        },
        "Sortino": {
            "Alpha One": 1.6,
            "Bravo One": 1.2,
            "Charlie One": 1.1,
        },
        "InformationRatio": {
            "Alpha One": 1.4,
            "Bravo One": 1.0,
            "Charlie One": 0.9,
        },
        "MaxDrawdown": {
            "Alpha One": -0.10,
            "Bravo One": -0.08,
            "Charlie One": -0.07,
        },
    }

    from trend_analysis.core import rank_selection as rank_mod

    def fake_metric_series(frame: pd.DataFrame, metric: str, _cfg: Any) -> pd.Series:
        vals = metrics[metric]
        return pd.Series({col: vals[col] for col in frame.columns}, dtype=float)

    monkeypatch.setattr(rank_mod, "_compute_metric_series", fake_metric_series)

    selector = ScriptedSelector(["Alpha One", "Bravo One", "Charlie One"])
    from trend_analysis import selector as selector_mod

    monkeypatch.setattr(
        selector_mod, "create_selector_by_name", lambda *a, **k: selector
    )

    weighting = SequenceWeighting(
        [{"Alpha One": 0.01, "Bravo One": 0.02, "Charlie One": 0.03}]
    )
    monkeypatch.setattr(mp_engine, "EqualWeight", lambda: weighting)
    monkeypatch.setattr(mp_engine, "Rebalancer", lambda *_: IdentityRebalancer())

    call_log: list[Dict[str, Any]] = []
    monkeypatch.setattr(mp_engine, "_run_analysis", _stub_run_analysis(call_log))

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 1
    assert call_log, "_run_analysis should receive custom weights"
    weights = call_log[0]["custom_weights"]
    assert set(weights) == {"Alpha One", "Bravo One", "Charlie One"}
    assert pytest.approx(sum(weights.values()), rel=1e-9) == 120.0


def test_threshold_hold_turnover_cap_scales_then_bounds(monkeypatch):
    """Turnover scaling fires before bounds re-normalise the portfolio."""

    cfg = DummyConfig()
    cfg.portfolio["threshold_hold"].update({"target_n": 2, "metric": "Sharpe"})
    cfg.portfolio["constraints"].update(
        {
            "max_funds": 2,
            "min_weight": 0.05,
            "max_weight": 0.3,
            "min_weight_strikes": 1,
        }
    )
    cfg.portfolio.update({"transaction_cost_bps": 25.0, "max_turnover": 0.2})

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
            "Alpha One": [0.06, 0.05, 0.04, 0.03],
            "Bravo One": [0.01, 0.02, 0.01, 0.02],
            "Charlie One": [0.03, 0.025, 0.02, 0.03],
            "Cash": [0.0, 0.0, 0.0, 0.0],
        }
    )

    periods = [
        DummyPeriod("2020-01-31", "2020-03-31", "2020-04-30", "2020-04-30"),
    ]

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)

    metrics = {
        "AnnualReturn": {
            "Alpha One": 0.20,
            "Bravo One": 0.05,
            "Charlie One": 0.18,
        },
        "Volatility": {
            "Alpha One": 0.18,
            "Bravo One": 0.12,
            "Charlie One": 0.15,
        },
        "Sharpe": {
            "Alpha One": 1.7,
            "Bravo One": 0.6,
            "Charlie One": 1.4,
        },
        "Sortino": {
            "Alpha One": 1.8,
            "Bravo One": 0.7,
            "Charlie One": 1.45,
        },
        "InformationRatio": {
            "Alpha One": 1.5,
            "Bravo One": 0.5,
            "Charlie One": 1.2,
        },
        "MaxDrawdown": {
            "Alpha One": -0.09,
            "Bravo One": -0.04,
            "Charlie One": -0.08,
        },
    }

    from trend_analysis.core import rank_selection as rank_mod

    def fake_metric_series(frame: pd.DataFrame, metric: str, _cfg: Any) -> pd.Series:
        vals = metrics[metric]
        return pd.Series({col: vals[col] for col in frame.columns}, dtype=float)

    monkeypatch.setattr(rank_mod, "_compute_metric_series", fake_metric_series)

    selector = ScriptedSelector(["Alpha One", "Bravo One", "Charlie One"])
    from trend_analysis import selector as selector_mod

    monkeypatch.setattr(
        selector_mod, "create_selector_by_name", lambda *a, **k: selector
    )

    weighting = SequenceWeighting(
        [
            {"Alpha One": 0.9, "Bravo One": 0.01, "Charlie One": 0.05},
            {"Alpha One": 0.9, "Bravo One": 0.01, "Charlie One": 0.05},
            {"Alpha One": 0.9, "Charlie One": 0.9},
        ]
    )
    monkeypatch.setattr(mp_engine, "EqualWeight", lambda: weighting)
    monkeypatch.setattr(mp_engine, "Rebalancer", lambda *_: IdentityRebalancer())

    call_log: list[Dict[str, Any]] = []
    monkeypatch.setattr(mp_engine, "_run_analysis", _stub_run_analysis(call_log))

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 1
    weights = call_log[0]["custom_weights"]
    assert pytest.approx(sum(weights.values()), rel=1e-9) == 60.0
    assert results[0]["turnover"] == pytest.approx(0.6)
    assert results[0]["transaction_cost"] == pytest.approx(0.0015, rel=1e-9)
