from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pandas as pd
import pytest

from trend_analysis.multi_period import engine as mp_engine


@dataclass
class ThresholdConfig:
    multi_period: Dict[str, Any] = field(
        default_factory=lambda: {
            "frequency": "M",
            "in_sample_len": 3,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-06",
        }
    )
    data: Dict[str, Any] = field(
        default_factory=lambda: {
            "csv_path": "unused.csv",
            "risk_free_column": "RF",
        }
    )
    portfolio: Dict[str, Any] = field(
        default_factory=lambda: {
            "policy": "threshold_hold",
            "transaction_cost_bps": 25.0,
            "max_turnover": 0.05,
            "threshold_hold": {
                "target_n": 4,
                "metric": "Sharpe",
                "z_exit_soft": -0.4,
                "z_entry_soft": 0.5,
                "min_weight_strikes": 2,
            },
            "constraints": {
                "max_funds": 4,
                "min_weight": 0.25,
                "max_weight": 0.6,
                "min_weight_strikes": 2,
            },
            "weighting": {"name": "adaptive", "params": {}},
            "indices_list": None,
        }
    )
    vol_adjust: Dict[str, Any] = field(default_factory=lambda: {"target_vol": 1.0})
    benchmarks: Dict[str, Any] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=lambda: {"monthly_cost": 0.0})
    performance: Dict[str, Any] = field(default_factory=dict)
    seed: int = 21

    def model_dump(self) -> Dict[str, Any]:
        return {
            "multi_period": dict(self.multi_period),
            "portfolio": dict(self.portfolio),
            "vol_adjust": dict(self.vol_adjust),
        }


@dataclass
class ShortConfig:
    multi_period: Dict[str, Any] = field(
        default_factory=lambda: {
            "frequency": "M",
            "in_sample_len": 2,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-04",
        }
    )
    data: Dict[str, Any] = field(
        default_factory=lambda: {
            "csv_path": "unused.csv",
            "risk_free_column": "RF",
        }
    )
    portfolio: Dict[str, Any] = field(
        default_factory=lambda: {
            "policy": "threshold_hold",
            "transaction_cost_bps": 0.0,
            "max_turnover": 1.0,
            "threshold_hold": {
                "target_n": 2,
                "metric": "Sharpe",
            },
            "constraints": {
                "max_funds": 2,
                "min_weight": 0.1,
                "max_weight": 0.35,
            },
            "weighting": {"name": "adaptive", "params": {}},
            "indices_list": None,
        }
    )
    vol_adjust: Dict[str, Any] = field(default_factory=lambda: {"target_vol": 1.0})
    benchmarks: Dict[str, Any] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=lambda: {"monthly_cost": 0.0})
    performance: Dict[str, Any] = field(default_factory=dict)
    seed: int = 99

    def model_dump(self) -> Dict[str, Any]:
        return {
            "multi_period": dict(self.multi_period),
            "portfolio": dict(self.portfolio),
            "vol_adjust": dict(self.vol_adjust),
        }


class SequencedWeighting:
    def __init__(self, sequences: List[Dict[str, float]]) -> None:
        self.sequences = sequences
        self.calls = 0
        self.updates: List[Tuple[pd.Series, int]] = []

    def weight(self, selected: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        idx = min(self.calls, len(self.sequences) - 1)
        seq = self.sequences[idx]
        self.calls += 1
        weights = pd.Series(
            {name: seq.get(name, 0.02) for name in selected.index},
            index=selected.index,
            dtype=float,
        )
        return weights.to_frame("weight")

    def update(self, scores: pd.Series, days: int) -> None:
        self.updates.append((scores.copy(), days))


class ScriptedSelector:
    def __init__(self, order: List[str], top_n: int) -> None:
        self._order = order
        self.top_n = top_n
        self.rank_column = "Sharpe"

    def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        ordered = score_frame.reindex(self._order).dropna(how="all")
        selected = ordered.head(self.top_n)
        return selected, selected


class StaticRebalancer:
    def __init__(self, *_cfg: Any) -> None:
        self.calls = 0

    def apply_triggers(
        self, prev_weights: pd.Series, _sf: pd.DataFrame, **kwargs
    ) -> pd.Series:
        self.calls += 1
        mapping = prev_weights.to_dict()
        if self.calls == 1:
            return pd.Series(
                {
                    "Alpha One": float(mapping.get("Alpha One", 0.3)),
                    "Alpha Two": 0.12,
                    "Gamma One": float(mapping.get("Gamma One", 0.05)),
                    "Delta One": 0.28,
                },
                dtype=float,
            )
        return prev_weights.astype(float)


class NoOpRebalancer:
    def __init__(self, *_cfg: Any) -> None:
        pass

    def apply_triggers(
        self, prev_weights: pd.Series, _sf: pd.DataFrame, **kwargs
    ) -> pd.Series:
        return prev_weights.astype(float)


def _build_base_frame() -> pd.DataFrame:
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
    data = {
        "Date": dates,
        "Alpha One": [0.05, 0.04, 0.03, 0.02, 0.01, 0.02],
        "Alpha Two": [0.04, 0.03, 0.02, 0.03, 0.02, 0.01],
        "Beta One": [0.01, 0.015, 0.02, 0.018, 0.017, 0.016],
        "Gamma One": [0.02, 0.021, 0.02, 0.015, 0.014, 0.013],
        "Delta One": [0.03, 0.032, 0.031, 0.029, 0.028, 0.027],
        "Epsilon One": [0.025, 0.026, 0.027, 0.028, 0.029, 0.03],
        "RF": [0.0] * len(dates),
    }
    return pd.DataFrame(data)


def _patch_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    import trend_analysis.core.rank_selection as rank_sel

    metric_maps: Dict[str, Dict[str, float]] = {
        "AnnualReturn": {
            "Alpha One": 0.11,
            "Alpha Two": 0.09,
            "Beta One": 0.03,
            "Gamma One": 0.01,
            "Delta One": 0.14,
            "Epsilon One": 0.06,
        },
        "Volatility": {
            "Alpha One": 0.2,
            "Alpha Two": 0.18,
            "Beta One": 0.12,
            "Gamma One": 0.1,
            "Delta One": 0.25,
            "Epsilon One": 0.15,
        },
        "Sharpe": {
            "Alpha One": 2.0,
            "Alpha Two": 1.8,
            "Beta One": -0.2,
            "Gamma One": -1.0,
            "Delta One": 3.0,
            "Epsilon One": 0.5,
        },
        "Sortino": {
            "Alpha One": 1.5,
            "Alpha Two": 1.3,
            "Beta One": 0.2,
            "Gamma One": 0.1,
            "Delta One": 2.0,
            "Epsilon One": 0.8,
        },
        "InformationRatio": {
            "Alpha One": 0.9,
            "Alpha Two": 0.8,
            "Beta One": 0.1,
            "Gamma One": -0.05,
            "Delta One": 1.3,
            "Epsilon One": 0.4,
        },
        "MaxDrawdown": {
            "Alpha One": -0.12,
            "Alpha Two": -0.11,
            "Beta One": -0.04,
            "Gamma One": -0.09,
            "Delta One": -0.2,
            "Epsilon One": -0.06,
        },
    }

    def fake_metric_series(frame: pd.DataFrame, metric: str, _cfg: Any) -> pd.Series:
        series = pd.Series(metric_maps[metric], dtype=float)
        return series.reindex(frame.columns).astype(float)

    monkeypatch.setattr(rank_sel, "_compute_metric_series", fake_metric_series)


def test_threshold_hold_event_log_and_replacements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ThresholdConfig()
    df = _build_base_frame()

    periods = [
        types.SimpleNamespace(
            in_start="2020-01-31",
            in_end="2020-03-31",
            out_start="2020-04-30",
            out_end="2020-04-30",
        ),
        types.SimpleNamespace(
            in_start="2020-02-29",
            in_end="2020-04-30",
            out_start="2020-05-31",
            out_end="2020-05-31",
        ),
    ]

    monkeypatch.setattr(mp_engine, "generate_periods", lambda *_: periods)
    _patch_metrics(monkeypatch)

    order = [
        "Alpha One",
        "Beta One",
        "Gamma One",
        "Delta One",
        "Alpha Two",
        "Epsilon One",
    ]

    monkeypatch.setattr(
        mp_engine,
        "AdaptiveBayesWeighting",
        lambda *args, **kwargs: SequencedWeighting(
            [
                {
                    "Alpha One": 0.7,
                    "Beta One": 0.1,
                    "Gamma One": 0.05,
                    "Delta One": 0.05,
                },
                {
                    "Alpha One": 0.8,
                    "Beta One": 0.04,
                    "Gamma One": 0.02,
                    "Delta One": 0.01,
                },
                {"Alpha One": 0.7, "Gamma One": 0.05, "Delta One": 0.2},
                {
                    "Alpha One": 0.7,
                    "Delta One": 0.2,
                    "Epsilon One": 0.1,
                    "Beta One": 0.05,
                },
            ]
        ),
    )

    monkeypatch.setattr(
        mp_engine,
        "Rebalancer",
        StaticRebalancer,
    )

    import trend_analysis.selector as selector_mod

    monkeypatch.setattr(
        selector_mod,
        "create_selector_by_name",
        lambda *_args, **_kwargs: ScriptedSelector(
            order, cfg.portfolio["threshold_hold"]["target_n"]
        ),
    )

    analysis_calls: List[Dict[str, Any]] = []

    def fake_run_analysis(
        _df: pd.DataFrame,
        in_start: str,
        in_end: str,
        out_start: str,
        out_end: str,
        _target_vol: float,
        _monthly_cost: float,
        *,
        custom_weights: Dict[str, float],
        manual_funds: List[str],
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        analysis_calls.append(
            {
                "period": (in_start, out_end),
                "weights": dict(custom_weights),
                "funds": list(manual_funds),
            }
        )
        return {"out_user_stats": {}, "out_ew_stats": {}}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2
    assert len(analysis_calls) == 2

    second_period = results[1]
    reasons = {event["reason"] for event in second_period["manager_changes"]}
    assert {
        "z_exit",
        "z_entry",
        "one_per_firm",
        "low_weight_strikes",
        "replacement",
    }.issubset(reasons)

    low_weight_events = [
        event
        for event in second_period["manager_changes"]
        if event["reason"] == "low_weight_strikes"
    ]
    assert low_weight_events and "below min" in low_weight_events[0]["detail"]

    weights_second = analysis_calls[1]["weights"]
    manual_funds = analysis_calls[1]["funds"]
    manual_total = sum(weights_second[f] for f in manual_funds)
    assert pytest.approx(manual_total, rel=1e-9) == 100.0
    assert set(manual_funds) == {"Alpha One", "Delta One", "Epsilon One", "Beta One"}
    gamma_weight = weights_second.get("Gamma One")
    assert gamma_weight == pytest.approx(
        cfg.portfolio["constraints"]["min_weight"] * 100.0, rel=1e-9
    )
    assert "Gamma One" not in manual_funds

    assert second_period["turnover"] == pytest.approx(0.25)
    expected_cost = second_period["turnover"] * (
        cfg.portfolio["transaction_cost_bps"] / 10000.0
    )
    assert second_period["transaction_cost"] == pytest.approx(expected_cost)


def test_threshold_hold_weight_bounds_deficit(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ShortConfig()
    df = _build_base_frame()

    periods = [
        types.SimpleNamespace(
            in_start="2020-01-31",
            in_end="2020-02-29",
            out_start="2020-03-31",
            out_end="2020-03-31",
        )
    ]

    monkeypatch.setattr(mp_engine, "generate_periods", lambda *_: periods)
    _patch_metrics(monkeypatch)

    monkeypatch.setattr(
        mp_engine,
        "AdaptiveBayesWeighting",
        lambda *args, **kwargs: SequencedWeighting(
            [
                {"Alpha One": 0.4, "Beta One": 0.3},
                {"Alpha One": 0.01, "Beta One": 0.01},
            ]
        ),
    )

    monkeypatch.setattr(mp_engine, "Rebalancer", NoOpRebalancer)

    import trend_analysis.selector as selector_mod

    monkeypatch.setattr(
        selector_mod,
        "create_selector_by_name",
        lambda *_args, **_kwargs: ScriptedSelector(["Alpha One", "Beta One"], 2),
    )

    analysis_calls: List[Dict[str, Any]] = []

    def fake_run_analysis(
        _df: pd.DataFrame,
        in_start: str,
        in_end: str,
        out_start: str,
        out_end: str,
        _target_vol: float,
        _monthly_cost: float,
        *,
        custom_weights: Dict[str, float],
        manual_funds: List[str],
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        analysis_calls.append(
            {
                "period": (in_start, out_end),
                "weights": dict(custom_weights),
                "funds": list(manual_funds),
            }
        )
        return {"out_user_stats": {}, "out_ew_stats": {}}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 1
    assert len(analysis_calls) == 1
    weights = analysis_calls[0]["weights"]
    assert pytest.approx(weights["Alpha One"], rel=1e-9) == 35.0
    assert pytest.approx(weights["Beta One"], rel=1e-9) == 35.0
    assert results[0]["turnover"] == pytest.approx(0.7)
    seed_events = [e for e in results[0]["manager_changes"] if e["reason"] == "seed"]
    assert len(seed_events) == 2
