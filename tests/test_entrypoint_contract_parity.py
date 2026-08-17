"""Regression coverage for configuration values shared by both analysis paths."""

from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis import api
from trend_analysis.config import Config
from trend_analysis.config.model import CostModelSettings, PortfolioSettings
from trend_analysis.config_contract import (
    resolve_pipeline_monthly_cost,
    resolve_portfolio_cost_bps,
    resolve_portfolio_weighting_name,
)
from trend_analysis.multi_period import run as run_mp
from trend_analysis.multi_period.engine import (
    _resolve_pipeline_monthly_cost,
    _resolve_portfolio_cost_bps,
    _resolve_portfolio_weighting,
)
from trend_analysis.pipeline_entrypoints import (
    _resolve_single_period_monthly_cost,
    _resolve_single_period_weighting_scheme,
)


def _parity_returns_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame({"Date": dates, "RF": 0.0, "A": 0.01, "B": 0.02, "C": 0.005})


def _single_period_cfg(portfolio: dict[str, object]) -> Config:
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
        portfolio=dict(portfolio),
        metrics={},
        export={},
        run={},
    )


def _multi_period_cfg(portfolio: dict[str, object]) -> Config:
    portfolio_cfg = dict(portfolio)
    portfolio_cfg.setdefault("policy", "threshold_hold")
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
        portfolio=portfolio_cfg,
        metrics={},
        export={},
        run={},
        multi_period={
            "frequency": "M",
            "in_sample_len": 2,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-06",
        },
    )


def test_same_config_same_numbers_across_entrypoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same portfolio config yields matching costs and weighting on both paths."""

    portfolio = {
        "cost_model": {"per_trade_bps": 12, "half_spread_bps": 3},
        "weighting": {"name": "score_prop_bayes"},
    }
    run = {"monthly_cost": 0.0}
    returns = _parity_returns_frame()

    single_cost = _resolve_single_period_monthly_cost(portfolio, run)
    multi_tc_bps, multi_half_spread_bps = _resolve_portfolio_cost_bps(portfolio)
    multi_cost = _resolve_pipeline_monthly_cost(
        run,
        portfolio,
    )
    _, _, _, _, multi_weighting = _resolve_portfolio_weighting(portfolio)

    assert single_cost == multi_cost == pytest.approx(0.0015)
    assert _resolve_single_period_weighting_scheme(portfolio, dict.get) == multi_weighting

    gross_return = 0.01
    zero_cost = _resolve_single_period_monthly_cost(
        {"cost_model": {"per_trade_bps": 0, "half_spread_bps": 0}}, run
    )
    assert gross_return - single_cost < gross_return - zero_cost
    assert gross_return - multi_cost < gross_return - zero_cost

    zero_portfolio = dict(portfolio)
    zero_portfolio["cost_model"] = {"per_trade_bps": 0.0, "half_spread_bps": 0.0}

    single_zero = api.run_simulation(_single_period_cfg(zero_portfolio), returns)
    single_charged = api.run_simulation(_single_period_cfg(portfolio), returns)
    assert single_charged.portfolio is not None
    assert single_zero.portfolio is not None
    assert single_charged.portfolio.mean() < single_zero.portfolio.mean()

    mp_zero = run_mp(_multi_period_cfg(zero_portfolio), returns.copy())
    mp_charged = run_mp(_multi_period_cfg(portfolio), returns.copy())
    assert mp_charged, "multi-period path returned no periods"
    charged_costs = [res["transaction_cost"] for res in mp_charged if "transaction_cost" in res]
    zero_costs = [res["transaction_cost"] for res in mp_zero if "transaction_cost" in res]
    assert charged_costs and zero_costs
    assert sum(charged_costs) > sum(zero_costs)

    weight_portfolio = {
        "weighting": {"name": "hrp"},
        "cost_model": {"per_trade_bps": 0.0, "half_spread_bps": 0.0},
    }
    captured: dict[str, object] = {}

    def fake_single_run(*args: object, **kwargs: object) -> dict[str, object]:
        captured["single_weighting_scheme"] = kwargs.get("weighting_scheme")
        return {
            "out_sample_stats": {},
            "benchmark_ir": {},
            "score_frame": pd.DataFrame(),
        }

    monkeypatch.setattr(api, "_run_analysis", fake_single_run)
    api.run_simulation(_single_period_cfg(weight_portfolio), returns)
    assert captured["single_weighting_scheme"] == "hrp"

    _, _, _, _, mp_scheme = _resolve_portfolio_weighting(weight_portfolio)
    assert mp_scheme == "hrp"


def test_typed_portfolio_cost_model_preserves_canonical_values() -> None:
    portfolio = PortfolioSettings(
        rebalance_calendar="NYSE",
        max_turnover=1.0,
        cost_model=CostModelSettings(per_trade_bps=12, half_spread_bps=3),
    )

    assert resolve_portfolio_cost_bps(portfolio) == (12.0, 3.0)
    assert _resolve_single_period_monthly_cost(portfolio, {"monthly_cost": 0.0}) == pytest.approx(
        0.0015
    )


def test_empty_cost_model_keeps_run_monthly_cost() -> None:
    """An empty mapping is not a configured portfolio cost override."""

    portfolio = {"cost_model": {}}
    run = {"monthly_cost": 0.0025}

    assert resolve_pipeline_monthly_cost(run, portfolio) == pytest.approx(0.0025)
    assert _resolve_single_period_monthly_cost(portfolio, run) == pytest.approx(0.0025)


@pytest.mark.parametrize(
    ("portfolio", "expected"),
    [
        ({"weighting": {"name": "ew"}}, "equal"),
        ({"weighting": {"name": "robust"}}, "robust_mv"),
        ({"weighting": {"name": "score_prop_bayes"}}, "score_prop_bayes"),
    ],
)
def test_weighting_names_match_both_entrypoints(
    portfolio: dict[str, object], expected: str
) -> None:
    """Canonical nested names share one resolution rule."""

    _, _, _, _, multi_weighting = _resolve_portfolio_weighting(portfolio)
    assert resolve_portfolio_weighting_name(portfolio) == expected
    assert _resolve_single_period_weighting_scheme(portfolio, dict.get) == expected
    assert multi_weighting == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_costs_are_rejected_by_shared_contract(value: float) -> None:
    """Both entry points reject non-finite and negative cost inputs consistently."""

    portfolio = {"cost_model": {"per_trade_bps": value}}
    with pytest.raises(Exception, match="cost_model.per_trade_bps"):
        _resolve_single_period_monthly_cost(portfolio, {"monthly_cost": 0.0})
