from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest

from trend.config_schema import CoreConfigError
from trend_analysis.multi_period import engine as mp_engine


@dataclass
class CostModelConfig:
    multi_period: dict[str, Any] = field(
        default_factory=lambda: {
            "frequency": "M",
            "in_sample_len": 2,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-04",
        }
    )
    data: dict[str, Any] = field(
        default_factory=lambda: {
            "csv_path": "unused.csv",
            "risk_free_column": "RF",
        }
    )
    portfolio: dict[str, Any] = field(
        default_factory=lambda: {
            "policy": "threshold_hold",
            "transaction_cost_bps": 0.0,
            "cost_model": {"bps_per_trade": 40.0, "slippage_bps": 10.0},
            "max_turnover": 1.0,
            "threshold_hold": {"target_n": 2, "metric": "Sharpe"},
            "constraints": {"max_funds": 2, "min_weight": 0.1, "max_weight": 0.35},
            "weighting": {"name": "adaptive", "params": {}},
            "indices_list": None,
        }
    )
    vol_adjust: dict[str, Any] = field(default_factory=lambda: {"target_vol": 1.0})
    benchmarks: dict[str, Any] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=lambda: {"monthly_cost": 0.0})
    performance: dict[str, Any] = field(default_factory=dict)
    seed: int = 99

    def model_dump(self) -> dict[str, Any]:
        return {
            "multi_period": dict(self.multi_period),
            "portfolio": dict(self.portfolio),
            "vol_adjust": dict(self.vol_adjust),
        }


class StaticWeighting:
    def weight(self, selected: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        del date
        weights = pd.Series(
            {name: 0.35 for name in selected.index},
            index=selected.index,
            dtype=float,
        )
        return weights.to_frame("weight")

    def update(self, scores: pd.Series, days: int) -> None:
        del scores, days


class StaticSelector:
    top_n = 2
    rank_column = "Sharpe"

    def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        selected = score_frame.reindex(["Alpha", "Beta"]).dropna(how="all")
        return selected, selected


class NoOpRebalancer:
    def __init__(self, *_cfg: Any) -> None:
        pass

    def apply_triggers(self, prev_weights: pd.Series, _sf: pd.DataFrame, **kwargs: Any) -> pd.Series:
        del _sf, kwargs
        return prev_weights.astype(float)


def _returns_frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    return pd.DataFrame(
        {
            "Date": dates,
            "Alpha": [0.03, 0.02, 0.01],
            "Beta": [0.02, 0.025, 0.015],
            "RF": [0.0, 0.0, 0.0],
        }
    )


def test_cost_model_bps_feed_multi_period_transaction_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = CostModelConfig()
    periods = [
        types.SimpleNamespace(
            in_start="2020-01-31",
            in_end="2020-02-29",
            out_start="2020-03-31",
            out_end="2020-03-31",
        )
    ]
    monkeypatch.setattr(mp_engine, "generate_periods", lambda *_: periods)
    monkeypatch.setattr(mp_engine, "AdaptiveBayesWeighting", lambda *_, **__: StaticWeighting())
    monkeypatch.setattr(mp_engine, "Rebalancer", NoOpRebalancer)

    import trend_analysis.core.rank_selection as rank_selection
    import trend_analysis.selector as selector_mod

    def metric_series(frame: pd.DataFrame, metric: str, _cfg: Any) -> pd.Series:
        values = {
            "AnnualReturn": {"Alpha": 0.1, "Beta": 0.08},
            "Volatility": {"Alpha": 0.2, "Beta": 0.18},
            "Sharpe": {"Alpha": 1.5, "Beta": 1.2},
            "Sortino": {"Alpha": 1.4, "Beta": 1.1},
            "InformationRatio": {"Alpha": 0.7, "Beta": 0.5},
            "MaxDrawdown": {"Alpha": -0.1, "Beta": -0.08},
        }
        return pd.Series(values[metric], dtype=float).reindex(frame.columns)

    monkeypatch.setattr(rank_selection, "_compute_metric_series", metric_series)
    monkeypatch.setattr(selector_mod, "create_selector_by_name", lambda *_args, **_kwargs: StaticSelector())
    monkeypatch.setattr(
        mp_engine,
        "_run_analysis",
        lambda *_args, **_kwargs: {"out_user_stats": {}, "out_ew_stats": {}},
    )

    results = mp_engine.run(cfg, df=_returns_frame())

    assert len(results) == 1
    assert results[0]["turnover"] == pytest.approx(0.7)
    assert results[0]["transaction_cost"] == pytest.approx(0.7 * 50.0 / 10000.0)


def test_cost_model_bps_feed_non_threshold_pipeline_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = CostModelConfig()
    cfg.portfolio["policy"] = "rank"
    periods = [
        types.SimpleNamespace(
            in_start="2020-01-31",
            in_end="2020-02-29",
            out_start="2020-03-31",
            out_end="2020-03-31",
        )
    ]
    monthly_costs: list[float] = []

    monkeypatch.setattr(mp_engine, "generate_periods", lambda *_: periods)

    def fake_run_analysis(*args: Any, **_kwargs: Any) -> dict[str, Any]:
        monthly_costs.append(float(args[6]))
        return {"out_user_stats": {}, "out_ew_stats": {}}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    results = mp_engine.run(cfg, df=_returns_frame())

    assert len(results) == 1
    assert monthly_costs == [pytest.approx(50.0 / 10000.0)]


def test_invalid_cost_model_bps_raises_core_config_error() -> None:
    with pytest.raises(CoreConfigError, match="portfolio.cost_model.bps_per_trade cannot be negative"):
        mp_engine._resolve_portfolio_cost_bps(
            {"cost_model": {"bps_per_trade": -1.0, "slippage_bps": 0.0}}
        )
