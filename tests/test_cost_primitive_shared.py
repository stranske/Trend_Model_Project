from __future__ import annotations

import pytest

from trend_analysis.backtesting.harness import CostModel
from trend_analysis.metrics.turnover import linear_turnover_cost
from trend_analysis.multi_period import engine
from trend_analysis.rebalancing.strategies import TurnoverCapStrategy


def test_linear_transaction_cost_sites_share_canonical_primitive() -> None:
    turnover = 0.42
    trade_bps = 17.0
    slippage_bps = 3.0
    combined_bps = trade_bps + slippage_bps
    expected = linear_turnover_cost(turnover, combined_bps)

    harness_model = CostModel(bps_per_trade=trade_bps, slippage_bps=slippage_bps)
    rebalancer = TurnoverCapStrategy({"cost_bps": combined_bps})

    assert harness_model.apply(turnover) == pytest.approx(expected)
    assert rebalancer._calculate_cost(turnover) == pytest.approx(expected)
    assert engine.linear_turnover_cost(turnover, combined_bps) == pytest.approx(expected)


def test_linear_transaction_cost_zero_or_negative_turnover_is_zero() -> None:
    assert linear_turnover_cost(0.0, 25.0) == 0.0
    assert linear_turnover_cost(-0.1, 25.0) == 0.0
