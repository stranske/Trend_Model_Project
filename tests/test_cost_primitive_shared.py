import pandas as pd
import pytest

from trend_analysis.backtesting.harness import CostModel
from trend_analysis.metrics.turnover import (
    linear_turnover_cost,
    realized_turnover,
    turnover_cost,
)
from trend_analysis.multi_period.engine import _period_turnover_cost
from trend_analysis.rebalancing.strategies import TurnoverCapStrategy


def test_linear_cost_sites_share_canonical_primitive() -> None:
    turnover = 0.37
    trade_bps = 12.5
    slippage_bps = 2.5
    effective_bps = trade_bps + slippage_bps
    expected = linear_turnover_cost(turnover, effective_bps)

    assert CostModel(
        bps_per_trade=trade_bps,
        slippage_bps=slippage_bps,
    ).apply(turnover) == pytest.approx(expected)
    assert TurnoverCapStrategy({"cost_bps": effective_bps})._calculate_cost(
        turnover
    ) == pytest.approx(expected)
    assert _period_turnover_cost(
        turnover,
        tc_bps=trade_bps,
        slippage_bps=slippage_bps,
    ) == pytest.approx(expected)


def test_turnover_cost_series_uses_linear_primitive() -> None:
    weights = pd.DataFrame(
        {
            "A": [0.50, 0.25, 0.55],
            "B": [0.50, 0.75, 0.45],
        },
        index=pd.to_datetime(["2026-01-31", "2026-02-28", "2026-03-31"]),
    )
    cost_bps = 20.0

    costs = turnover_cost(weights, cost_bps=cost_bps)
    expected = realized_turnover(weights)["turnover"].map(
        lambda turnover: linear_turnover_cost(float(turnover), cost_bps)
    )

    pd.testing.assert_series_equal(costs, expected)
