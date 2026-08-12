"""Regression coverage for configuration values shared by both analysis paths."""

from __future__ import annotations

import pytest

from trend_analysis.config.model import CostModelSettings
from trend_analysis.config_contract import (
    resolve_pipeline_monthly_cost,
    resolve_portfolio_cost_bps,
    resolve_portfolio_weighting_name,
)
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
    zero_cost = _resolve_single_period_monthly_cost({"transaction_cost_bps": 0}, run)
    assert gross_return - single_cost < gross_return - zero_cost
    assert gross_return - multi_cost < gross_return - zero_cost


def test_cost_model_dump_preserves_legacy_values_when_optional_aliases_are_null() -> None:
    """Pydantic's null optional aliases must not mask configured legacy costs."""

    cost_model = CostModelSettings(bps_per_trade=12, slippage_bps=3).model_dump()
    portfolio = {"cost_model": cost_model}

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
        ({"weighting": {"name": "score_prop_bayes"}, "weighting_scheme": "robust"}, "robust_mv"),
        ({"weighting": {"name": "score_prop_bayes"}, "weighting_scheme": "equal"}, "score_prop_bayes"),
    ],
)
def test_weighting_aliases_and_precedence_match_both_entrypoints(
    portfolio: dict[str, object], expected: str
) -> None:
    """Nested aliases and explicit legacy weighting settings share one precedence rule."""

    _, _, _, _, multi_weighting = _resolve_portfolio_weighting(portfolio)
    assert resolve_portfolio_weighting_name(portfolio) == expected
    assert _resolve_single_period_weighting_scheme(portfolio, dict.get) == expected
    assert multi_weighting == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_costs_are_rejected_by_shared_contract(value: float) -> None:
    """Both entry points reject non-finite and negative cost inputs consistently."""

    portfolio = {"cost_model": {"bps_per_trade": value}}
    with pytest.raises(Exception, match="cost_model.bps_per_trade"):
        _resolve_single_period_monthly_cost(portfolio, {"monthly_cost": 0.0})
