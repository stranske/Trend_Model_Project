"""Execution-level tests for the single-period transaction-cost contract."""

from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis import api
from trend_analysis.config import Config


def _make_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame({"Date": dates, "RF": 0.0, "A": 0.01, "B": 0.012})


def _make_single_period_cfg(transaction_cost_bps: float | None) -> Config:
    portfolio: dict[str, object] = {}
    if transaction_cost_bps is not None:
        portfolio["cost_model"] = {
            "per_trade_bps": transaction_cost_bps,
            "half_spread_bps": 0,
        }
    return Config(
        version="1",
        data={
            "risk_free_column": "RF",
            "allow_risk_free_fallback": False,
            "date_column": "Date",
            "frequency": "M",
        },
        preprocessing={},
        vol_adjust={"target_vol": 1.0},
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-03",
            "out_start": "2020-04",
            "out_end": "2020-06",
        },
        portfolio=portfolio,
        metrics={},
        export={},
        run={},
    )


def test_single_period_transaction_cost_bps_reduces_net_returns() -> None:
    """The public single-period API charges configured portfolio costs."""
    df = _make_df()
    zero_cost = api.run_simulation(_make_single_period_cfg(0.0), df)
    charged = api.run_simulation(_make_single_period_cfg(25.0), df)

    assert charged.portfolio is not None
    assert zero_cost.portfolio is not None
    assert charged.portfolio.mean() < zero_cost.portfolio.mean()


def test_single_period_api_passes_nested_weighting_name_to_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public API resolves nested weighting names before pipeline dispatch."""
    cfg = _make_single_period_cfg(0.0)
    cfg.portfolio = {
        "weighting": {
            "name": "score_prop_bayes",
            "params": {"column": "Sortino", "shrink_tau": 0.5},
        }
    }
    captured: dict[str, object] = {}

    def fake_run_analysis(*args: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "out_sample_stats": {},
            "benchmark_ir": {},
            "score_frame": pd.DataFrame(),
        }

    monkeypatch.setattr(api, "_run_analysis_with_diagnostics", fake_run_analysis)
    api.run_simulation(cfg, _make_df())

    assert captured["weighting_scheme"] == "score_prop_bayes"
    assert captured["weight_engine_params"] == {
        "column": "Sortino",
        "shrink_tau": 0.5,
    }


def test_single_period_api_forwards_registered_engine_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructor parameters are not limited to built-in score engines."""
    cfg = _make_single_period_cfg(0.0)
    cfg.portfolio = {
        "weighting": {
            "name": "third_party_weight_engine",
            "params": {"scale": 2.5},
        }
    }
    captured: dict[str, object] = {}

    def fake_run_analysis(*args: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "out_sample_stats": {},
            "benchmark_ir": {},
            "score_frame": pd.DataFrame(),
        }

    monkeypatch.setattr(api, "_run_analysis_with_diagnostics", fake_run_analysis)
    api.run_simulation(cfg, _make_df())

    assert captured["weighting_scheme"] == "third_party_weight_engine"
    assert captured["weight_engine_params"] == {"scale": 2.5}
