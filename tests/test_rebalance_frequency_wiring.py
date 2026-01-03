from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trend_analysis.multi_period import engine as mp_engine


@dataclass
class RebalanceConfig:
    multi_period: dict[str, object] = field(
        default_factory=lambda: {
            "frequency": "A",
            "in_sample_len": 1,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2021-12",
        }
    )
    data: dict[str, object] = field(
        default_factory=lambda: {
            "csv_path": "unused.csv",
            "risk_free_column": "RF",
        }
    )
    portfolio: dict[str, object] = field(
        default_factory=lambda: {
            "policy": "threshold_hold",
            "selection_mode": "rank",
            "rebalance_freq": "M",
            "threshold_hold": {"target_n": 2, "metric": "Sharpe"},
            "constraints": {
                "max_funds": 3,
                "min_weight": 0.05,
                "max_weight": 0.8,
            },
            "rank": {"inclusion_approach": "top_n"},
            "weighting_scheme": "equal",
            "weighting": {"name": "score_prop_bayes", "params": {"shrink_tau": 0.25}},
        }
    )
    vol_adjust: dict[str, object] = field(
        default_factory=lambda: {"enabled": True, "target_vol": 0.1}
    )
    benchmarks: dict[str, object] = field(default_factory=dict)
    run: dict[str, object] = field(default_factory=lambda: {"monthly_cost": 0.0})
    performance: dict[str, object] = field(default_factory=dict)
    seed: int = 7

    def model_dump(self) -> dict[str, object]:
        return {
            "multi_period": dict(self.multi_period),
            "portfolio": dict(self.portfolio),
            "vol_adjust": dict(self.vol_adjust),
        }


def _sample_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    idx = np.arange(len(dates), dtype=float)
    fund_a = 0.01 + 0.02 * np.sin(idx / 3.0)
    fund_b = 0.015 + 0.05 * np.sin(idx / 2.0 + 0.3)
    fund_c = 0.005 + 0.03 * np.cos(idx / 4.0)
    rf = np.full(len(dates), 0.001)
    return pd.DataFrame(
        {
            "Date": dates,
            "FundA": fund_a,
            "FundB": fund_b,
            "FundC": fund_c,
            "RF": rf,
        }
    )


def _combined_user_series(results: list[dict[str, object]]) -> pd.Series:
    series = [
        res["portfolio_user_weight"]
        for res in results
        if isinstance(res.get("portfolio_user_weight"), pd.Series)
    ]
    if not series:
        raise AssertionError("Expected portfolio_user_weight series in results.")
    return pd.concat(series).sort_index()


def test_rebalance_frequency_changes_returns() -> None:
    cfg = RebalanceConfig()
    monthly_results = mp_engine.run(cfg, _sample_returns())
    monthly_series = _combined_user_series(monthly_results)

    cfg.portfolio["rebalance_freq"] = "A"
    annual_results = mp_engine.run(cfg, _sample_returns())
    annual_series = _combined_user_series(annual_results)

    assert monthly_series.index.equals(annual_series.index)
    assert not np.allclose(monthly_series.values, annual_series.values)

    monthly_weights = monthly_results[0].get("weights_user_weight")
    annual_weights = annual_results[0].get("weights_user_weight")
    assert isinstance(monthly_weights, pd.DataFrame)
    assert isinstance(annual_weights, pd.DataFrame)
    assert len(monthly_weights.index) > len(annual_weights.index)
