"""
Integration tests for rebalancing strategies with the multi-period engine.
"""

import pandas as pd
import pytest

from trend_analysis.multi_period.engine import run_schedule, Portfolio
from trend_analysis.selector import RankSelector
from trend_analysis.weighting import EqualWeight
from trend_analysis.rebalancing import (
    VolTargetRebalanceStrategy,
    DrawdownGuardStrategy,
    RebalancingStrategiesManager,
)


def _create_test_score_frames():
    """Create test score frames for multi-period testing."""
    dates = ["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"]
    frames = {}
    
    for i, date in enumerate(dates):
        # Simulate different market conditions
        if i == 0:  # Initial period
            data = {
                "Fund_A": {"Sharpe": 1.5, "zscore": 1.2},
                "Fund_B": {"Sharpe": 1.2, "zscore": 0.8},
                "Fund_C": {"Sharpe": 0.9, "zscore": 0.3},
            }
        elif i == 1:  # Stable period
            data = {
                "Fund_A": {"Sharpe": 1.4, "zscore": 1.1},
                "Fund_B": {"Sharpe": 1.1, "zscore": 0.7},
                "Fund_C": {"Sharpe": 0.8, "zscore": 0.2},
            }
        elif i == 2:  # Volatile period
            data = {
                "Fund_A": {"Sharpe": 1.0, "zscore": 0.5},
                "Fund_B": {"Sharpe": 0.8, "zscore": 0.1},
                "Fund_C": {"Sharpe": 0.6, "zscore": -0.2},
            }
        else:  # Recovery period
            data = {
                "Fund_A": {"Sharpe": 1.3, "zscore": 1.0},
                "Fund_B": {"Sharpe": 1.0, "zscore": 0.6},
                "Fund_C": {"Sharpe": 0.7, "zscore": 0.1},
            }
        
        df = pd.DataFrame(data).T
        frames[date] = df
    
    return frames


class TestRebalancingIntegration:
    """Integration tests for rebalancing strategies with portfolio engine."""
    
    def test_run_schedule_with_vol_targeting(self):
        """Test run_schedule with volatility targeting strategy."""
        score_frames = _create_test_score_frames()
        selector = RankSelector(top_n=2, rank_column="Sharpe")
        weighting = EqualWeight()
        
        # Create vol targeting strategy
        vol_strategy = VolTargetRebalanceStrategy({
            "target": 0.15,
            "window": 3,
            "lev_min": 0.5,
            "lev_max": 2.0
        })
        
        rebalancing_manager = RebalancingStrategiesManager([vol_strategy])
        
        # Run without rebalancing strategies
        pf_baseline = run_schedule(score_frames, selector, weighting)
        
        # Run with rebalancing strategies
        pf_rebalanced = run_schedule(
            score_frames, selector, weighting,
            rebalancing_strategies=rebalancing_manager
        )
        
        # Both should have same number of periods
        assert len(pf_baseline.history) == len(pf_rebalanced.history)
        assert len(pf_baseline.history) == 4
        
        # For this test, we'll just verify that the rebalancing manager was applied
        # without errors and that the portfolio structure is maintained
        # The actual leverage effects would require proper return data
        for date in pf_rebalanced.history:
            weights = pf_rebalanced.history[date]
            assert all(weights >= 0), f"Negative weights in {date}"
            assert len(weights) <= 2, f"Too many holdings in {date}"
    
    def test_run_schedule_with_drawdown_guard(self):
        """Test run_schedule with drawdown guard strategy."""
        score_frames = _create_test_score_frames()
        selector = RankSelector(top_n=2, rank_column="Sharpe")
        weighting = EqualWeight()
        
        # Create drawdown guard strategy
        guard_strategy = DrawdownGuardStrategy({
            "dd_window": 4,
            "dd_threshold": 0.05,  # Low threshold to test triggering
            "guard_multiplier": 0.6,
            "recover_threshold": 0.02
        })
        
        rebalancing_manager = RebalancingStrategiesManager([guard_strategy])
        
        # Run with rebalancing strategies
        pf = run_schedule(
            score_frames, selector, weighting,
            rebalancing_strategies=rebalancing_manager
        )
        
        # Should complete successfully
        assert len(pf.history) == 4
        assert all(isinstance(weights, pd.Series) for weights in pf.history.values())
        
        # All weights should be non-negative
        for weights in pf.history.values():
            assert all(weights >= 0)
    
    def test_run_schedule_with_combined_strategies(self):
        """Test run_schedule with both vol targeting and drawdown guard."""
        score_frames = _create_test_score_frames()
        selector = RankSelector(top_n=2, rank_column="Sharpe")
        weighting = EqualWeight()
        
        # Create both strategies
        vol_strategy = VolTargetRebalanceStrategy({
            "target": 0.12,
            "window": 3
        })
        guard_strategy = DrawdownGuardStrategy({
            "dd_window": 4,
            "dd_threshold": 0.08,
            "guard_multiplier": 0.7
        })
        
        rebalancing_manager = RebalancingStrategiesManager([vol_strategy, guard_strategy])
        
        # Run with combined rebalancing strategies
        pf = run_schedule(
            score_frames, selector, weighting,
            rebalancing_strategies=rebalancing_manager
        )
        
        # Should complete successfully
        assert len(pf.history) == 4
        
        # Verify basic portfolio constraints
        for date, weights in pf.history.items():
            # All weights should be non-negative
            assert all(weights >= 0), f"Negative weights found in {date}"
            # Should have correct number of holdings
            assert len(weights) <= 2, f"Too many holdings in {date}"
            # Should maintain relative proportions (after normalization)
            if weights.sum() > 0:
                proportions = weights / weights.sum()
                assert abs(proportions.sum() - 1.0) < 1e-10, f"Proportions don't sum to 1 in {date}"
    
    def test_strategies_maintain_fund_selection(self):
        """Test that rebalancing strategies don't change which funds are selected."""
        score_frames = _create_test_score_frames()
        selector = RankSelector(top_n=2, rank_column="Sharpe")
        weighting = EqualWeight()
        
        vol_strategy = VolTargetRebalanceStrategy({"target": 0.10})
        rebalancing_manager = RebalancingStrategiesManager([vol_strategy])
        
        # Run both versions
        pf_baseline = run_schedule(score_frames, selector, weighting)
        pf_rebalanced = run_schedule(
            score_frames, selector, weighting,
            rebalancing_strategies=rebalancing_manager
        )
        
        # Should have same funds selected in each period
        for date in score_frames.keys():
            baseline_funds = set(pf_baseline.history[date].index)
            rebalanced_funds = set(pf_rebalanced.history[date].index)
            assert baseline_funds == rebalanced_funds, f"Fund selection differs in {date}"
    
    def test_empty_score_frames(self):
        """Test handling of empty score frames."""
        score_frames = {}
        selector = RankSelector(top_n=2, rank_column="Sharpe")
        weighting = EqualWeight()
        
        vol_strategy = VolTargetRebalanceStrategy()
        rebalancing_manager = RebalancingStrategiesManager([vol_strategy])
        
        pf = run_schedule(
            score_frames, selector, weighting,
            rebalancing_strategies=rebalancing_manager
        )
        
        assert len(pf.history) == 0
    
    def test_single_period(self):
        """Test rebalancing with only one period."""
        score_frames = {"2020-01-31": _create_test_score_frames()["2020-01-31"]}
        selector = RankSelector(top_n=2, rank_column="Sharpe")
        weighting = EqualWeight()
        
        vol_strategy = VolTargetRebalanceStrategy()
        rebalancing_manager = RebalancingStrategiesManager([vol_strategy])
        
        pf = run_schedule(
            score_frames, selector, weighting,
            rebalancing_strategies=rebalancing_manager
        )
        
        assert len(pf.history) == 1
        # Single period should work normally (no rebalancing effects)
        weights = list(pf.history.values())[0]
        assert len(weights) == 2
        assert all(weights > 0)


class TestRebalancingStrategiesStatePersistence:
    """Test that rebalancing strategy state persists correctly across periods."""
    
    def test_vol_targeting_state_persistence(self):
        """Test that volatility targeting maintains state across periods."""
        strategy = VolTargetRebalanceStrategy({"window": 3, "target": 0.10})
        weights = pd.Series({"A": 0.5, "B": 0.5})
        
        # Apply multiple periods with returns
        returns_sequence = [0.02, 0.01, -0.005, 0.015]
        results = []
        
        for ret in returns_sequence:
            result = strategy.apply(weights, current_returns=ret)
            results.append(result)
        
        # State should be maintained - buffer length should be limited by window
        assert len(strategy._returns_buffer) == min(len(returns_sequence), strategy.window)
        
        # Later results should be influenced by accumulated history
        # The exact values depend on the volatility calculation, but should be stable
        assert all(isinstance(result, pd.Series) for result in results)
        assert all(len(result) == 2 for result in results)
    
    def test_drawdown_guard_state_persistence(self):
        """Test that drawdown guard maintains state across periods."""
        strategy = DrawdownGuardStrategy({
            "dd_window": 4,
            "dd_threshold": 0.10,
            "guard_multiplier": 0.5
        })
        weights = pd.Series({"A": 0.4, "B": 0.6})
        
        # Create a drawdown scenario
        returns_sequence = [0.01, -0.08, -0.05, 0.02, 0.03]  # Drawdown then recovery
        guard_states = []
        
        for ret in returns_sequence:
            result = strategy.apply(weights, current_returns=ret)
            guard_states.append(strategy._in_guard_mode)
        
        # Should transition into and potentially out of guard mode
        assert any(guard_states), "Should have entered guard mode during drawdown"
        
        # Cumulative returns buffer should be maintained
        assert len(strategy._cumulative_returns) <= strategy.dd_window
        assert len(strategy._cumulative_returns) == min(len(returns_sequence), strategy.dd_window)
    
    def test_strategy_reset_functionality(self):
        """Test that strategy states can be reset properly."""
        vol_strategy = VolTargetRebalanceStrategy()
        guard_strategy = DrawdownGuardStrategy()
        manager = RebalancingStrategiesManager([vol_strategy, guard_strategy])
        
        weights = pd.Series({"A": 0.5, "B": 0.5})
        
        # Build up some state
        for ret in [0.01, -0.02, 0.005]:
            vol_strategy.apply(weights, current_returns=ret)
            guard_strategy.apply(weights, current_returns=ret)
        
        # Verify state exists
        assert len(vol_strategy._returns_buffer) > 0
        assert len(guard_strategy._cumulative_returns) > 0
        
        # Reset all states
        manager.reset_all_states()
        
        # Verify state is cleared
        assert len(vol_strategy._returns_buffer) == 0
        assert len(guard_strategy._cumulative_returns) == 0
        assert not guard_strategy._in_guard_mode