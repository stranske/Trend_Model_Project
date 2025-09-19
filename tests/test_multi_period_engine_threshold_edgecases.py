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
    data: Dict[str, Any] = field(default_factory=lambda: {"csv_path": "unused.csv"})
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

    def weight(self, selected: pd.DataFrame) -> pd.DataFrame:
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

    def apply_triggers(self, prev_weights: pd.Series, _sf: pd.DataFrame) -> pd.Series:
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

    periods = [
        DummyPeriod("2020-01-31", "2020-02-29", "2020-03-31", "2020-03-31")
    ]

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
            {"Alpha One": 0.55, "Alpha Two": 0.25, "Beta One": 0.15, "Gamma One": 0.04, "Delta One": 0.01},
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
    assert run_calls[1]["custom_weights"] == {
        "Alpha One": pytest.approx(32.5, rel=1e-3),
        "Beta One": pytest.approx(67.5, rel=1e-3),
    }
    assert results[1]["turnover"] == pytest.approx(0.4, rel=1e-6)
    assert results[1]["transaction_cost"] == pytest.approx(0.4 * 15.0 / 10000.0)


def test_run_schedule_applies_strategy_and_turnover_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score_frames = {
        "2020-01-31": pd.DataFrame({"Sharpe": [1.0, 0.5]}, index=["Alpha One", "Beta One"]),
        "2020-02-29": pd.DataFrame({"Sharpe": [0.3, 1.2]}, index=["Alpha One", "Beta One"]),
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
