"""Tests for turnover_cap rebalancing strategy."""

import pytest
import pandas as pd
from trend_analysis.rebalancing import (
    TurnoverCapStrategy,
    create_rebalancing_strategy,
    apply_rebalancing_strategies,
)


class TestTurnoverCapStrategy:
    """Test turnover cap rebalancing strategy."""

    def test_no_turnover_cap_needed(self):
        """Test when desired trades are within turnover limit."""
        strategy = TurnoverCapStrategy(
            {"max_turnover": 0.2, "cost_bps": 10, "priority": "largest_gap"}
        )

        current = pd.Series([0.3, 0.3, 0.4], index=["A", "B", "C"])
        target = pd.Series([0.35, 0.25, 0.4], index=["A", "B", "C"])

        new_weights, cost = strategy.apply(current, target)

        # Should execute all trades since total turnover = 0.1 < 0.2
        pd.testing.assert_series_equal(new_weights, target)
        expected_cost = 0.1 * 0.001  # 0.1 turnover * 10 bps
        assert abs(cost - expected_cost) < 1e-6

    def test_turnover_cap_enforcement(self):
        """Test turnover cap limits total trades."""
        strategy = TurnoverCapStrategy(
            {"max_turnover": 0.15, "cost_bps": 20, "priority": "largest_gap"}
        )

        current = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
        target = pd.Series([0.2, 0.6, 0.2], index=["A", "B", "C"])

        new_weights, cost = strategy.apply(current, target)

        # Total desired turnover = |0.2-0.5| + |0.6-0.3| + |0.2-0.2| = 0.6
        # Should be scaled to 0.15
        actual_turnover = (new_weights - current).abs().sum()
        assert abs(actual_turnover - 0.15) < 1e-6

        # Cost should be based on actual turnover
        expected_cost = 0.15 * 0.002  # 0.15 turnover * 20 bps
        assert abs(cost - expected_cost) < 1e-6

    def test_largest_gap_priority(self):
        """Test largest_gap prioritization."""
        strategy = TurnoverCapStrategy(
            {"max_turnover": 0.1, "cost_bps": 0, "priority": "largest_gap"}
        )

        current = pd.Series([0.4, 0.3, 0.3], index=["A", "B", "C"])
        target = pd.Series([0.1, 0.5, 0.4], index=["A", "B", "C"])

        new_weights, cost = strategy.apply(current, target)

        # Desired trades: A: -0.3, B: +0.2, C: +0.1
        # Largest gap is A (-0.3), then B (+0.2), then C (+0.1)
        # With 0.1 turnover budget, should execute A fully (0.3 turnover used)
        # No budget left for B or C

        # A should be reduced by 0.3/0.6 * 0.1 = 0.05 (scaled)
        # Actually, let's recalculate: A trade is 0.3 absolute, B is 0.2, C is 0.1
        # Total desired = 0.6, budget = 0.1, so scale factor = 0.1/0.6 = 1/6
        # But with prioritization, A gets full priority

        # A has largest gap (0.3), should get priority
        # If we have 0.1 budget and A needs 0.3, A gets 0.1/0.3 = 1/3 of its trade
        assert abs(new_weights["A"] - 0.3) < 1e-6
        assert new_weights["B"] == 0.3  # No change
        assert new_weights["C"] == 0.3  # No change

    def test_best_score_delta_priority(self):
        """Test best_score_delta prioritization."""
        strategy = TurnoverCapStrategy(
            {"max_turnover": 0.2, "cost_bps": 0, "priority": "best_score_delta"}
        )

        current = pd.Series([0.4, 0.3, 0.3], index=["A", "B", "C"])
        target = pd.Series([0.2, 0.5, 0.3], index=["A", "B", "C"])
        scores = pd.Series([1.0, 2.0, 0.5], index=["A", "B", "C"])

        new_weights, cost = strategy.apply(current, target, scores=scores)

        # Desired trades: A: -0.2 (score 1.0), B: +0.2 (score 2.0), C: 0
        # Score-weighted benefits: A: |-0.2 * 1.0| = 0.2, B: |+0.2 * 2.0| = 0.4
        # B should get higher priority due to higher score

        # Total desired turnover = 0.4, budget = 0.2
        # B has higher priority (0.4) vs A (0.2), so B gets executed first
        assert new_weights["B"] == 0.5  # B trade fully executed
        assert abs(new_weights["A"] - 0.4) < 1e-6  # A unchanged due to budget limit
        assert new_weights["C"] == 0.3  # C unchanged

    def test_fallback_to_largest_gap_when_no_scores(self):
        """Test fallback to largest_gap when scores not provided."""
        strategy = TurnoverCapStrategy(
            {"max_turnover": 0.1, "cost_bps": 0, "priority": "best_score_delta"}
        )

        current = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
        target = pd.Series([0.2, 0.6, 0.2], index=["A", "B", "C"])

        # No scores provided - should fall back to largest_gap
        new_weights, cost = strategy.apply(current, target)

        # Should behave same as largest_gap priority
        actual_turnover = (new_weights - current).abs().sum()
        assert abs(actual_turnover - 0.1) < 1e-6

    def test_zero_turnover_cap(self):
        """Test zero turnover cap prevents all trades."""
        strategy = TurnoverCapStrategy(
            {"max_turnover": 0.0, "cost_bps": 10, "priority": "largest_gap"}
        )

        current = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
        target = pd.Series([0.2, 0.6, 0.2], index=["A", "B", "C"])

        new_weights, cost = strategy.apply(current, target)

        # No trades should be executed
        pd.testing.assert_series_equal(new_weights, current)
        assert cost == 0.0

    def test_asset_index_alignment(self):
        """Test proper handling of different asset indices."""
        strategy = TurnoverCapStrategy(
            {"max_turnover": 0.5, "cost_bps": 0, "priority": "largest_gap"}
        )

        current = pd.Series([0.6, 0.4], index=["A", "B"])
        target = pd.Series([0.3, 0.3, 0.4], index=["A", "B", "C"])

        new_weights, cost = strategy.apply(current, target)

        # Should handle new asset C and align indices properly
        assert "C" in new_weights.index
        assert new_weights["C"] > 0  # C should get some allocation
        assert len(new_weights) == 3

    def test_high_transaction_costs(self):
        """Test transaction cost calculation."""
        strategy = TurnoverCapStrategy(
            {"max_turnover": 0.3, "cost_bps": 100, "priority": "largest_gap"}  # 1%
        )

        current = pd.Series([0.5, 0.5], index=["A", "B"])
        target = pd.Series([0.3, 0.7], index=["A", "B"])

        new_weights, cost = strategy.apply(current, target)

        # Total desired turnover = 0.4, but cap is 0.3, so actual turnover = 0.3
        expected_cost = 0.3 * 0.01  # 0.3 turnover * 100 bps
        assert abs(cost - expected_cost) < 1e-6

    def test_partial_trade_execution(self):
        """Test partial execution of final trade when budget runs out."""
        strategy = TurnoverCapStrategy(
            {"max_turnover": 0.25, "cost_bps": 0, "priority": "largest_gap"}
        )

        current = pd.Series([0.4, 0.3, 0.3], index=["A", "B", "C"])
        target = pd.Series([0.1, 0.6, 0.3], index=["A", "B", "C"])

        new_weights, cost = strategy.apply(current, target)

        # Desired trades: A: -0.3, B: +0.3, C: 0
        # Both A and B have same magnitude, but only 0.25 budget available
        # Should execute first trade fully (0.3 used) but that exceeds budget
        # So should execute partial trades to fit budget

        actual_turnover = (new_weights - current).abs().sum()
        assert abs(actual_turnover - 0.25) < 1e-6

        # Should have moved toward target but not reached it
        assert new_weights["A"] < current["A"]  # Moved toward target
        assert new_weights["A"] > target["A"]  # But not fully there


class TestRebalancingIntegration:
    """Test integration of rebalancing strategies."""

    def test_create_strategy_by_name(self):
        """Test strategy creation by name."""
        strategy = create_rebalancing_strategy(
            "turnover_cap", {"max_turnover": 0.1, "cost_bps": 15}
        )

        assert isinstance(strategy, TurnoverCapStrategy)
        assert strategy.max_turnover == 0.1
        assert strategy.cost_bps == 15

    def test_unknown_strategy_raises_error(self):
        """Test error on unknown strategy."""

    with pytest.raises(ValueError, match="Unknown plugin"):
        create_rebalancing_strategy("unknown_strategy")

    def test_multiple_strategies_sequence(self):
        """Test applying multiple strategies in sequence."""
        strategies = ["drift_band", "turnover_cap"]
        params = {
            "drift_band": {"band_pct": 0.05, "mode": "partial"},
            "turnover_cap": {
                "max_turnover": 0.6,
                "cost_bps": 10,
            },  # High enough to not interfere
        }

        current = pd.Series([0.4, 0.3, 0.3], index=["A", "B", "C"])
        target = pd.Series([0.1, 0.6, 0.3], index=["A", "B", "C"])

        final_weights, total_cost = apply_rebalancing_strategies(
            strategies, params, current, target
        )

        # Should apply drift_band first, then turnover_cap
        assert isinstance(final_weights, pd.Series)
        assert total_cost >= 0

        # Since turnover cap is high (0.6), drift_band changes should be preserved
        actual_turnover = (final_weights - current).abs().sum()
        assert actual_turnover <= 0.6 + 1e-6  # Allow small numerical error


if __name__ == "__main__":
    pytest.main([__file__])
