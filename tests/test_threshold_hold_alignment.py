from pathlib import Path

import pandas as pd
import pytest
import yaml

from trend_analysis.config import Config
from trend_analysis.multi_period import engine as mp_engine
from trend_analysis.multi_period.scheduler import generate_periods


def test_threshold_hold_results_align_with_periods():
    cfg_data = yaml.safe_load(Path("config/defaults.yml").read_text())
    cfg_data["multi_period"] = {
        "frequency": "M",
        "in_sample_len": 2,
        "out_sample_len": 1,
        "start": "2020-01",
        "end": "2020-05",
    }
    portfolio = cfg_data.setdefault("portfolio", {})
    portfolio["policy"] = "threshold_hold"
    th_cfg = portfolio.setdefault("threshold_hold", {})
    th_cfg.update({"target_n": 3, "metric": "Sharpe"})
    constraints = portfolio.setdefault("constraints", {})
    constraints.update(
        {"max_funds": 4, "min_weight": 0.05, "max_weight": 0.6, "min_weight_strikes": 1}
    )
    weighting_cfg = portfolio.setdefault("weighting", {})
    weighting_cfg.update({"name": "adaptive_bayes", "params": {}})

    cfg = Config(**cfg_data)

    # Simple deterministic data set with enough funds
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "A Alpha": [0.05, 0.07, 0.06, 0.08, 0.07],
            "B Beta": [0.01, 0.005, 0.002, 0.001, 0.003],
            "C Capital": [0.03, 0.035, 0.04, 0.045, 0.05],
            "D Delta": [0.06, 0.07, 0.08, 0.09, 0.085],
            "E Echo": [0.025, 0.03, 0.028, 0.027, 0.026],
        }
    )

    # Monkeypatch scoring metrics to be deterministic without heavy computation
    import trend_analysis.core.rank_selection as rank_sel

    metric_maps = {
        "AnnualReturn": {
            "A Alpha": 0.12,
            "B Beta": 0.03,
            "C Capital": 0.18,
            "D Delta": 0.22,
            "E Echo": 0.2,
        },
        "Volatility": {
            "A Alpha": 0.25,
            "B Beta": 0.15,
            "C Capital": 0.2,
            "D Delta": 0.3,
            "E Echo": 0.18,
        },
        "Sharpe": {
            "A Alpha": 0.6,
            "B Beta": 0.1,
            "C Capital": 1.2,
            "D Delta": 1.5,
            "E Echo": 1.1,
        },
        "Sortino": {
            "A Alpha": 0.8,
            "B Beta": 0.2,
            "C Capital": 1.0,
            "D Delta": 1.6,
            "E Echo": 1.2,
        },
        "InformationRatio": {
            "A Alpha": 0.5,
            "B Beta": 0.05,
            "C Capital": 0.9,
            "D Delta": 1.3,
            "E Echo": 1.0,
        },
        "MaxDrawdown": {
            "A Alpha": -0.12,
            "B Beta": -0.05,
            "C Capital": -0.08,
            "D Delta": -0.1,
            "E Echo": -0.09,
        },
    }

    def fake_metric_series(_frame, metric, _stats_cfg):  # pragma: no cover - trivial
        return pd.Series(metric_maps[metric], dtype=float)

    monkeypatch_ctx = pytest.MonkeyPatch()
    monkeypatch_ctx.setattr(rank_sel, "_compute_metric_series", fake_metric_series)

    # Minimal stub for _run_analysis to avoid heavy pipeline work
    def fake_run_analysis(*_args, **_kwargs):
        return {"out_ew_stats": {"sharpe": 0.3}, "out_user_stats": {"sharpe": 0.4}}

    monkeypatch_ctx.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    results = mp_engine.run(cfg, df)
    periods = generate_periods(cfg.model_dump())
    assert len(results) == len(
        periods
    ), "Each generated period must produce a result entry"
    # Ensure each result has a period tuple of length 4
    for res in results:
        assert "period" in res and len(res["period"]) == 4

    # Clean up monkeypatch
    monkeypatch_ctx.undo()
