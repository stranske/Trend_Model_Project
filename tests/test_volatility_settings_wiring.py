from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trend_analysis.multi_period import engine as mp_engine


@dataclass
class VolatilityConfig:
    multi_period: dict[str, object] = field(
        default_factory=lambda: {
            "frequency": "M",
            "in_sample_len": 6,
            "out_sample_len": 2,
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
            "transaction_cost_bps": 0.0,
            "threshold_hold": {
                "target_n": 2,
                "metric": "Sharpe",
                "z_entry_soft": 0.5,
                "z_exit_soft": -0.5,
                "soft_strikes": 1,
                "entry_soft_strikes": 1,
            },
            "constraints": {
                "max_funds": 3,
                "min_weight": 0.05,
                "max_weight": 0.8,
            },
            "rank": {"inclusion_approach": "top_n"},
            "weighting_scheme": "equal",
        }
    )
    vol_adjust: dict[str, object] = field(
        default_factory=lambda: {
            "enabled": True,
            "target_vol": 0.1,
            "window": {"length": 6, "decay": "ewma", "lambda": 0.94},
        }
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


def _mean_scale_factor(results: list[dict[str, object]]) -> float:
    values: list[float] = []
    for result in results:
        diagnostics = result.get("risk_diagnostics", {})
        scale_factors = diagnostics.get("scale_factors") if diagnostics else None
        if scale_factors is None:
            continue
        if isinstance(scale_factors, pd.Series):
            values.append(float(scale_factors.mean()))
        else:
            values.append(float(pd.Series(scale_factors, dtype=float).mean()))
    if not values:
        raise AssertionError("No scale factor diagnostics were returned.")
    return float(np.mean(values))


def _run_config(cfg: VolatilityConfig) -> float:
    results = mp_engine.run(cfg, _sample_returns())
    assert results, "Expected multi-period results to be returned."
    return _mean_scale_factor(results)


def test_vol_adjust_enabled_changes_scaling() -> None:
    cfg = VolatilityConfig()
    enabled_mean = _run_config(cfg)
    cfg.vol_adjust["enabled"] = False
    disabled_mean = _run_config(cfg)
    assert not np.isclose(enabled_mean, disabled_mean, rtol=1e-4, atol=1e-6)
    assert np.isclose(disabled_mean, 1.0, rtol=1e-4, atol=1e-6)


def test_vol_window_length_changes_scaling() -> None:
    cfg = VolatilityConfig()
    cfg.vol_adjust["window"] = {"length": 6, "decay": "simple", "lambda": 0.94}
    baseline_mean = _run_config(cfg)
    cfg.vol_adjust["window"] = {"length": 3, "decay": "simple", "lambda": 0.94}
    test_mean = _run_config(cfg)
    assert not np.isclose(baseline_mean, test_mean, rtol=1e-4, atol=1e-6)


def test_vol_window_length_changes_scaling_for_ewma() -> None:
    cfg = VolatilityConfig()
    cfg.vol_adjust["window"] = {"length": 6, "decay": "ewma", "lambda": 0.9}
    baseline_mean = _run_config(cfg)
    cfg.vol_adjust["window"] = {"length": 3, "decay": "ewma", "lambda": 0.9}
    test_mean = _run_config(cfg)
    assert not np.isclose(baseline_mean, test_mean, rtol=1e-4, atol=1e-6)


def test_vol_window_decay_changes_scaling() -> None:
    cfg = VolatilityConfig()
    cfg.vol_adjust["window"] = {"length": 6, "decay": "ewma", "lambda": 0.9}
    baseline_mean = _run_config(cfg)
    cfg.vol_adjust["window"] = {"length": 6, "decay": "simple", "lambda": 0.9}
    test_mean = _run_config(cfg)
    assert not np.isclose(baseline_mean, test_mean, rtol=1e-4, atol=1e-6)


def test_vol_ewma_lambda_changes_scaling() -> None:
    cfg = VolatilityConfig()
    cfg.vol_adjust["window"] = {"length": 6, "decay": "ewma", "lambda": 0.9}
    baseline_mean = _run_config(cfg)
    cfg.vol_adjust["window"] = {"length": 6, "decay": "ewma", "lambda": 0.6}
    test_mean = _run_config(cfg)
    assert not np.isclose(baseline_mean, test_mean, rtol=1e-4, atol=1e-6)
