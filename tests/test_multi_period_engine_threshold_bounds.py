from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd
import pytest

from trend_analysis.multi_period import engine as mp_engine


@dataclass
class MinimalConfig:
    """Minimal configuration for exercising threshold-hold weight bounds."""

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
            "allow_risk_free_fallback": True,
        }
    )
    portfolio: Dict[str, Any] = field(
        default_factory=lambda: {
            "policy": "threshold_hold",
            "transaction_cost_bps": 0.0,
            "max_turnover": 1.0,
            "threshold_hold": {
                "target_n": 3,
                "metric": "Sharpe",
                "soft_strikes": 1,
                "entry_soft_strikes": 1,
                "z_exit_soft": -5.0,
                "z_entry_soft": -5.0,
            },
            "constraints": {
                "max_funds": 3,
                "min_weight": 0.2,
                "max_weight": 0.55,
                "min_weight_strikes": 1,
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


class ScriptedSelector:
    """Selector that keeps the provided score-frame ordering."""

    rank_column = "Sharpe"

    def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return score_frame, score_frame


class ScriptedWeighting:
    """Weighting sequence crafted to exercise bound adjustments."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.calls = 0
        self.sequences: List[Dict[str, float]] = [
            {"Alpha One": 0.6, "Alpha Two": 0.4, "Beta One": 0.2, "Gamma One": 0.1},
            {"Alpha One": 0.4, "Beta One": 0.1, "Gamma One": 0.05},
            {"Alpha One": 0.9, "Beta One": 0.4, "Gamma One": 0.35},
        ]

    def weight(self, selected: pd.DataFrame, date: pd.Timestamp | None = None) -> pd.DataFrame:
        del date
        seq = self.sequences[min(self.calls, len(self.sequences) - 1)]
        self.calls += 1
        weights = pd.Series(
            {idx: seq.get(idx, 0.05) for idx in selected.index},
            index=selected.index,
            dtype=float,
        )
        return weights.to_frame("weight")

    def update(self, scores: pd.Series, days: int) -> None:  # pragma: no cover - inert hook
        pass


class StaticRebalancer:
    """Rebalancer that preserves prior holdings for deterministic tests."""

    def __init__(self, *_cfg: Any) -> None:
        self.calls = 0

    def apply_triggers(self, prev_weights: pd.Series, _sf: pd.DataFrame, **kwargs) -> pd.Series:
        self.calls += 1
        return prev_weights.astype(float)


def test_threshold_hold_weight_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = MinimalConfig()

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
            "Alpha One": [0.05, 0.04, 0.03, 0.02, 0.01],
            "Alpha Two": [0.06, 0.05, 0.04, 0.03, 0.02],
            "Beta One": [0.02, 0.03, 0.02, 0.01, 0.02],
            "Gamma One": [0.04, 0.05, 0.06, 0.07, 0.08],
        }
    )

    periods = [
        DummyPeriod("2020-01-31", "2020-03-31", "2020-04-30", "2020-04-30"),
        DummyPeriod("2020-02-29", "2020-04-30", "2020-05-31", "2020-05-31"),
    ]

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)
    monkeypatch.setattr(mp_engine, "AdaptiveBayesWeighting", ScriptedWeighting)
    monkeypatch.setattr(mp_engine, "Rebalancer", StaticRebalancer)

    import trend_analysis.selector as selector_mod

    monkeypatch.setattr(selector_mod, "create_selector_by_name", lambda *a, **k: ScriptedSelector())

    import trend_analysis.core.rank_selection as rank_sel

    metric_maps = {
        "AnnualReturn": {
            "Alpha One": 0.15,
            "Alpha Two": 0.12,
            "Beta One": 0.07,
            "Gamma One": 0.18,
        },
        "Volatility": {
            "Alpha One": 0.25,
            "Alpha Two": 0.2,
            "Beta One": 0.15,
            "Gamma One": 0.3,
        },
        "Sharpe": {
            "Alpha One": 0.9,
            "Alpha Two": 0.8,
            "Beta One": 0.4,
            "Gamma One": 1.1,
        },
        "Sortino": {
            "Alpha One": 1.1,
            "Alpha Two": 0.9,
            "Beta One": 0.45,
            "Gamma One": 1.3,
        },
        "InformationRatio": {
            "Alpha One": 0.6,
            "Alpha Two": 0.5,
            "Beta One": 0.3,
            "Gamma One": 0.9,
        },
        "MaxDrawdown": {
            "Alpha One": -0.12,
            "Alpha Two": -0.11,
            "Beta One": -0.05,
            "Gamma One": -0.09,
        },
    }

    def fake_metric_series(_frame: pd.DataFrame, metric: str, _stats_cfg: Any) -> pd.Series:
        values = metric_maps[metric]
        return pd.Series(values, dtype=float)

    monkeypatch.setattr(rank_sel, "_compute_metric_series", fake_metric_series)

    records: List[Dict[str, Any]] = []

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
        records.append(
            {
                "weights": dict(custom_weights),
                "funds": list(manual_funds),
                "period": (in_start, out_end),
            }
        )
        return {"metrics": pd.DataFrame(), "details": {}, "seed": cfg.seed}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2

    # The scripted weighting sequences should normalise within the weight bounds.
    assert records
    first_weights = records[0]["weights"]
    assert set(first_weights) == {"Alpha One", "Beta One", "Gamma One"}
    assert pytest.approx(sum(first_weights.values()), rel=1e-9) == 100.0

    second_weights = records[1]["weights"]
    assert pytest.approx(sum(second_weights.values()), rel=1e-9) == 100.0
    assert set(records[0]["funds"]) == {"Alpha One", "Beta One", "Gamma One"}
    assert set(records[1]["funds"]) == {"Alpha One", "Beta One", "Gamma One"}


def test_threshold_hold_max_active_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = MinimalConfig()
    cfg.portfolio["constraints"]["max_active_positions"] = 2

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
            "Alpha One": [0.05, 0.04, 0.03, 0.02, 0.01],
            "Alpha Two": [0.06, 0.05, 0.04, 0.03, 0.02],
            "Beta One": [0.02, 0.03, 0.02, 0.01, 0.02],
            "Gamma One": [0.04, 0.05, 0.06, 0.07, 0.08],
        }
    )

    periods = [
        DummyPeriod("2020-01-31", "2020-03-31", "2020-04-30", "2020-04-30"),
        DummyPeriod("2020-02-29", "2020-04-30", "2020-05-31", "2020-05-31"),
    ]

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)
    monkeypatch.setattr(mp_engine, "AdaptiveBayesWeighting", ScriptedWeighting)
    monkeypatch.setattr(mp_engine, "Rebalancer", StaticRebalancer)

    import trend_analysis.selector as selector_mod

    monkeypatch.setattr(selector_mod, "create_selector_by_name", lambda *a, **k: ScriptedSelector())

    import trend_analysis.core.rank_selection as rank_sel

    metric_maps = {
        "AnnualReturn": {
            "Alpha One": 0.15,
            "Alpha Two": 0.12,
            "Beta One": 0.07,
            "Gamma One": 0.18,
        },
        "Volatility": {
            "Alpha One": 0.25,
            "Alpha Two": 0.2,
            "Beta One": 0.15,
            "Gamma One": 0.3,
        },
        "Sharpe": {
            "Alpha One": 0.9,
            "Alpha Two": 0.8,
            "Beta One": 0.4,
            "Gamma One": 1.1,
        },
        "Sortino": {
            "Alpha One": 1.1,
            "Alpha Two": 0.9,
            "Beta One": 0.45,
            "Gamma One": 1.3,
        },
        "InformationRatio": {
            "Alpha One": 0.6,
            "Alpha Two": 0.5,
            "Beta One": 0.3,
            "Gamma One": 0.9,
        },
        "MaxDrawdown": {
            "Alpha One": -0.12,
            "Alpha Two": -0.11,
            "Beta One": -0.05,
            "Gamma One": -0.09,
        },
    }

    def fake_metric_series(_frame: pd.DataFrame, metric: str, _stats_cfg: Any) -> pd.Series:
        values = metric_maps[metric]
        return pd.Series(values, dtype=float)

    monkeypatch.setattr(rank_sel, "_compute_metric_series", fake_metric_series)

    records: List[Dict[str, Any]] = []

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
        records.append(
            {
                "weights": dict(custom_weights),
                "funds": list(manual_funds),
                "period": (in_start, out_end),
            }
        )
        return {"metrics": pd.DataFrame(), "details": {}, "seed": cfg.seed}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2
    assert records
    for record in records:
        assert len(record["weights"]) <= 2
        assert len(record["funds"]) <= 2
