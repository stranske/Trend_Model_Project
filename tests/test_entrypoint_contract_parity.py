"""Regression coverage for configuration values shared by both analysis paths."""

from __future__ import annotations

import pytest

from trend_analysis.multi_period.engine import (
    _resolve_pipeline_monthly_cost,
    _resolve_portfolio_cost_bps,
    _resolve_portfolio_weighting,
)
from trend_analysis.pipeline_entrypoints import (
    _resolve_single_period_monthly_cost,
    _resolve_single_period_weighting_scheme,
)


def test_same_config_same_numbers_across_entrypoints() -> None:
    """Cost and weighting inputs agree before either entry point executes them."""

    portfolio = {
        "transaction_cost_bps": 12,
        "cost_model": {"slippage_bps": 3},
        "weighting": {"name": "score_prop_bayes"},
    }
    run = {"monthly_cost": 0.0}

    single_cost = _resolve_single_period_monthly_cost(portfolio, run)
    multi_tc_bps, multi_slippage_bps = _resolve_portfolio_cost_bps(portfolio)
    multi_cost = _resolve_pipeline_monthly_cost(
        run,
        portfolio,
        tc_bps=multi_tc_bps,
        slippage_bps=multi_slippage_bps,
    )
    _, _, _, _, multi_weighting = _resolve_portfolio_weighting(portfolio)

    assert single_cost == multi_cost == pytest.approx(0.0015)
    assert _resolve_single_period_weighting_scheme(portfolio, dict.get) == multi_weighting

    gross_return = 0.01
    zero_cost = _resolve_single_period_monthly_cost(
        {"transaction_cost_bps": 0}, run
    )
    assert gross_return - single_cost < gross_return - zero_cost
    assert gross_return - multi_cost < gross_return - zero_cost
