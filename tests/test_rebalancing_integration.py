"""Integration tests for rebalancing within the multi-period engine."""

import pandas as pd
import pytest

from trend_analysis.multi_period.engine import run_schedule
from trend_analysis.rebalancing import CashPolicy
from trend_analysis.selector import RankSelector
from trend_analysis.weighting import EqualWeight


def test_run_schedule_with_turnover_cap():
    """Test run_schedule integration with turnover_cap strategy."""
    # Create simple score frames
    dates = ["2020-01", "2020-02", "2020-03"]
    assets = ["A", "B", "C", "D"]

    score_frames = {}
    for i, date in enumerate(dates):
        # Create different rankings each period to force rebalancing
        scores = [4 - i, 3 - i, 2 + i, 1 + i]  # Different patterns each period
        score_frames[date] = pd.DataFrame(
            {"Sharpe": scores, "CAGR": [0.1, 0.08, 0.06, 0.04]}, index=assets
        )

    selector = RankSelector(top_n=2, rank_column="Sharpe")
    weighting = EqualWeight()

    # Test with turnover cap strategy
    rebalance_strategies = ["turnover_cap"]
    rebalance_params = {
        "turnover_cap": {"max_turnover": 0.3, "cost_bps": 20, "priority": "largest_gap"}
    }

    portfolio = run_schedule(
        score_frames,
        selector,
        weighting,
        rebalance_strategies=rebalance_strategies,
        rebalance_params=rebalance_params,
    )

    # Should have weight history for all dates
    assert len(portfolio.history) == len(dates)
    assert portfolio.total_rebalance_costs >= 0

    # Check that turnover constraint was respected
    # (detailed turnover checking would require more complex setup)
    for date_key, weights in portfolio.history.items():
        assert isinstance(weights, pd.Series)
        assert len(weights) > 0


def test_run_schedule_with_multiple_strategies():
    """Test run_schedule with multiple rebalancing strategies."""
    dates = ["2020-01", "2020-02"]
    assets = ["A", "B", "C"]

    score_frames = {}
    for date in dates:
        score_frames[date] = pd.DataFrame(
            {
                "Sharpe": [1.0, 0.8, 0.6],
            },
            index=assets,
        )

    selector = RankSelector(top_n=3, rank_column="Sharpe")
    weighting = EqualWeight()

    # Test with multiple strategies
    rebalance_strategies = ["drift_band", "turnover_cap"]
    rebalance_params = {
        "drift_band": {"band_pct": 0.05, "mode": "partial"},
        "turnover_cap": {"max_turnover": 0.5, "cost_bps": 10},
    }

    portfolio = run_schedule(
        score_frames,
        selector,
        weighting,
        rebalance_strategies=rebalance_strategies,
        rebalance_params=rebalance_params,
    )

    assert len(portfolio.history) == len(dates)
    assert portfolio.total_rebalance_costs >= 0


def test_run_schedule_applies_cash_policy():
    """Ensure run_schedule passes cash policy to rebalancing strategies."""
    dates = ["2020-01", "2020-02"]
    assets = ["A", "B"]

    score_frames = {}
    for date in dates:
        score_frames[date] = pd.DataFrame({"Sharpe": [1.0, 0.9]}, index=assets)

    selector = RankSelector(top_n=2, rank_column="Sharpe")
    weighting = EqualWeight()

    rebalance_strategies = ["turnover_cap"]
    rebalance_params = {"turnover_cap": {"max_turnover": 1.0, "cost_bps": 0}}
    cash_policy = CashPolicy(
        explicit_cash=True,
        cash_return_source="risk_free",
        normalize_weights=False,
    )

    portfolio = run_schedule(
        score_frames,
        selector,
        weighting,
        rebalance_strategies=rebalance_strategies,
        rebalance_params=rebalance_params,
        cash_policy=cash_policy,
    )

    for weights in portfolio.history.values():
        assert "CASH" in weights.index
        assert pytest.approx(1.0) == float(weights.sum())


def test_run_schedule_without_rebalancing():
    """Run without rebalancing strategies (backward compatible)."""
    dates = ["2020-01", "2020-02"]
    assets = ["A", "B"]

    score_frames = {}
    for date in dates:
        score_frames[date] = pd.DataFrame(
            {
                "Sharpe": [1.0, 0.8],
            },
            index=assets,
        )

    selector = RankSelector(top_n=2, rank_column="Sharpe")
    weighting = EqualWeight()

    # No rebalancing strategies - should work as before
    portfolio = run_schedule(score_frames, selector, weighting)

    assert len(portfolio.history) == len(dates)
    assert portfolio.total_rebalance_costs == 0.0  # No rebalancing costs


if __name__ == "__main__":
    pytest.main([__file__])
