"""
Tests for rebalancing strategies.
"""

import numpy as np
import pandas as pd
import pytest

from trend_analysis.rebalancing import (
    VolTargetRebalanceStrategy,
    DrawdownGuardStrategy,
    RebalancingStrategiesManager,
    create_rebalancing_strategy,
)


class TestVolTargetRebalanceStrategy:
    """Tests for volatility targeting strategy."""
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        strategy = VolTargetRebalanceStrategy()
        assert strategy.target == 0.10
        assert strategy.window == 6
        assert strategy.estimator == "simple"
        assert strategy.lev_min == 0.5
        assert strategy.lev_max == 1.5
        assert strategy._returns_buffer == []
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        params = {
            "target": 0.15,
            "window": 10,
            "estimator": "ewma",
            "lev_min": 0.3,
            "lev_max": 2.0
        }
        strategy = VolTargetRebalanceStrategy(params)
        assert strategy.target == 0.15
        assert strategy.window == 10
        assert strategy.estimator == "ewma"
        assert strategy.lev_min == 0.3
        assert strategy.lev_max == 2.0
    
    def test_apply_empty_weights(self):
        """Test applying strategy to empty weights."""
        strategy = VolTargetRebalanceStrategy()
        weights = pd.Series([], dtype=float)
        result = strategy.apply(weights)
        assert result.empty
    
    def test_apply_insufficient_returns(self):
        """Test that strategy returns original weights without sufficient returns history."""
        strategy = VolTargetRebalanceStrategy()
        weights = pd.Series({"A": 0.5, "B": 0.5})
        
        # No returns - should return original weights
        result = strategy.apply(weights)
        pd.testing.assert_series_equal(result, weights)
        
        # Only one return - should still return original weights
        result = strategy.apply(weights, current_returns=0.01)
        pd.testing.assert_series_equal(result, weights)
    
    def test_apply_vol_targeting_simple(self):
        """Test volatility targeting with simple estimator."""
        strategy = VolTargetRebalanceStrategy({"target": 0.10, "window": 3})
        weights = pd.Series({"A": 0.6, "B": 0.4})
        
        # Add returns to build up history (high volatility scenario)
        returns = [0.05, -0.03, 0.04, -0.02]  # High volatility
        for ret in returns:
            result = strategy.apply(weights, current_returns=ret)
        
        # Should reduce leverage due to high volatility
        assert result.sum() < weights.sum()
        # But should still maintain proportions
        expected_props = weights / weights.sum()
        actual_props = result / result.sum()
        pd.testing.assert_series_equal(actual_props, expected_props, rtol=1e-10)
    
    def test_apply_vol_targeting_ewma(self):
        """Test volatility targeting with EWMA estimator."""
        strategy = VolTargetRebalanceStrategy({
            "target": 0.10,
            "window": 4,
            "estimator": "ewma"
        })
        weights = pd.Series({"A": 0.5, "B": 0.5})
        
        # Add returns to build up history
        returns = [0.005, 0.003, 0.002, 0.001]  # Very low volatility
        for ret in returns:
            result = strategy.apply(weights, current_returns=ret)
        
        # Should increase leverage due to low volatility (but capped at lev_max)
        # Note: for very low volatility, should hit max leverage
        leverage = result.sum() / weights.sum()
        assert leverage >= 1.0  # Should at least maintain or increase leverage
    
    def test_leverage_bounds(self):
        """Test that leverage is properly bounded."""
        strategy = VolTargetRebalanceStrategy({
            "target": 0.10,
            "lev_min": 0.8,
            "lev_max": 1.2
        })
        weights = pd.Series({"A": 0.5, "B": 0.5})
        
        # Very high volatility scenario - should hit minimum leverage
        high_vol_returns = [0.10, -0.08, 0.12, -0.09]
        for ret in high_vol_returns:
            result = strategy.apply(weights, current_returns=ret)
        
        leverage = result.sum() / weights.sum()
        assert leverage >= strategy.lev_min - 1e-10
        
        # Reset and test maximum leverage
        strategy.reset_state()
        # Very low volatility scenario - should hit maximum leverage
        low_vol_returns = [0.001, 0.0005, 0.0008, -0.0002]
        for ret in low_vol_returns:
            result = strategy.apply(weights, current_returns=ret)
            
        leverage = result.sum() / weights.sum()
        assert leverage <= strategy.lev_max + 1e-10
    
    def test_zero_volatility(self):
        """Test behavior with zero volatility."""
        strategy = VolTargetRebalanceStrategy()
        weights = pd.Series({"A": 0.5, "B": 0.5})
        
        # Zero volatility returns
        zero_returns = [0.0, 0.0, 0.0]
        for ret in zero_returns:
            result = strategy.apply(weights, current_returns=ret)
        
        # Should return original weights when volatility is zero
        pd.testing.assert_series_equal(result, weights)
    
    def test_reset_state(self):
        """Test state reset functionality."""
        strategy = VolTargetRebalanceStrategy()
        weights = pd.Series({"A": 0.5, "B": 0.5})
        
        # Build up some returns history
        for ret in [0.01, 0.02, -0.01]:
            strategy.apply(weights, current_returns=ret)
        
        assert len(strategy._returns_buffer) > 0
        
        # Reset state
        strategy.reset_state()
        assert len(strategy._returns_buffer) == 0


class TestDrawdownGuardStrategy:
    """Tests for drawdown guard strategy."""
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        strategy = DrawdownGuardStrategy()
        assert strategy.dd_window == 12
        assert strategy.dd_threshold == 0.10
        assert strategy.guard_multiplier == 0.5
        assert strategy.recover_threshold == 0.05
        assert strategy._cumulative_returns == []
        assert strategy._in_guard_mode is False
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        params = {
            "dd_window": 6,
            "dd_threshold": 0.15,
            "guard_multiplier": 0.3,
            "recover_threshold": 0.03
        }
        strategy = DrawdownGuardStrategy(params)
        assert strategy.dd_window == 6
        assert strategy.dd_threshold == 0.15
        assert strategy.guard_multiplier == 0.3
        assert strategy.recover_threshold == 0.03
    
    def test_apply_empty_weights(self):
        """Test applying strategy to empty weights."""
        strategy = DrawdownGuardStrategy()
        weights = pd.Series([], dtype=float)
        result = strategy.apply(weights)
        assert result.empty
    
    def test_apply_insufficient_returns(self):
        """Test that strategy returns original weights without sufficient returns history."""
        strategy = DrawdownGuardStrategy()
        weights = pd.Series({"A": 0.5, "B": 0.5})
        
        # No returns - should return original weights
        result = strategy.apply(weights)
        pd.testing.assert_series_equal(result, weights)
        
        # Only one return - should still return original weights
        result = strategy.apply(weights, current_returns=0.01)
        pd.testing.assert_series_equal(result, weights)
    
    def test_no_drawdown_scenario(self):
        """Test behavior with no significant drawdown."""
        strategy = DrawdownGuardStrategy({"dd_threshold": 0.10})
        weights = pd.Series({"A": 0.6, "B": 0.4})
        
        # Positive returns - no drawdown
        positive_returns = [0.02, 0.01, 0.03, 0.005]
        for ret in positive_returns:
            result = strategy.apply(weights, current_returns=ret)
        
        # Should not be in guard mode
        assert not strategy._in_guard_mode
        # Should return original weights
        pd.testing.assert_series_equal(result, weights)
    
    def test_drawdown_trigger_guard_mode(self):
        """Test that significant drawdown triggers guard mode."""
        strategy = DrawdownGuardStrategy({
            "dd_threshold": 0.08,
            "guard_multiplier": 0.4
        })
        weights = pd.Series({"A": 0.5, "B": 0.5})
        
        # Create a drawdown scenario
        returns_sequence = [0.01, -0.05, -0.04, -0.02]  # Creates significant drawdown
        
        for ret in returns_sequence:
            result = strategy.apply(weights, current_returns=ret)
        
        # Should be in guard mode after significant drawdown
        assert strategy._in_guard_mode
        # Weights should be reduced by guard multiplier
        expected = weights * strategy.guard_multiplier
        pd.testing.assert_series_equal(result, expected)
    
    def test_recovery_from_guard_mode(self):
        """Test recovery from guard mode when drawdown improves."""
        strategy = DrawdownGuardStrategy({
            "dd_threshold": 0.08,
            "guard_multiplier": 0.5,
            "recover_threshold": 0.03,
            "dd_window": 8
        })
        weights = pd.Series({"A": 0.5, "B": 0.5})
        
        # First create a drawdown to trigger guard mode
        bad_returns = [0.01, -0.06, -0.04, -0.02]
        for ret in bad_returns:
            strategy.apply(weights, current_returns=ret)
        
        assert strategy._in_guard_mode
        
        # Now recover with good returns
        good_returns = [0.05, 0.04, 0.03, 0.02]
        for ret in good_returns:
            result = strategy.apply(weights, current_returns=ret)
        
        # Should exit guard mode when drawdown recovers below threshold
        assert not strategy._in_guard_mode
        # Should return full weights
        pd.testing.assert_series_equal(result, weights)
    
    def test_rolling_window_behavior(self):
        """Test that the strategy respects the rolling window for drawdown calculation."""
        window = 4
        strategy = DrawdownGuardStrategy({
            "dd_window": window,
            "dd_threshold": 0.08
        })
        weights = pd.Series({"A": 0.5, "B": 0.5})
        
        # Create returns longer than window
        returns = [0.01, -0.05, -0.04, -0.02, 0.03, 0.02, 0.01]
        
        for ret in returns:
            strategy.apply(weights, current_returns=ret)
        
        # Should only keep the last 'window' cumulative returns
        assert len(strategy._cumulative_returns) == window
    
    def test_reset_state(self):
        """Test state reset functionality."""
        strategy = DrawdownGuardStrategy()
        weights = pd.Series({"A": 0.5, "B": 0.5})
        
        # Build up some state
        for ret in [0.01, -0.05, -0.03]:
            strategy.apply(weights, current_returns=ret)
        strategy._in_guard_mode = True
        
        assert len(strategy._cumulative_returns) > 0
        assert strategy._in_guard_mode
        
        # Reset state
        strategy.reset_state()
        assert len(strategy._cumulative_returns) == 0
        assert not strategy._in_guard_mode


class TestRebalancingStrategiesManager:
    """Tests for the strategies manager."""
    
    def test_init_empty(self):
        """Test initialization without strategies."""
        manager = RebalancingStrategiesManager()
        assert len(manager.strategies) == 0
    
    def test_init_with_strategies(self):
        """Test initialization with strategies."""
        strategies = [
            VolTargetRebalanceStrategy(),
            DrawdownGuardStrategy()
        ]
        manager = RebalancingStrategiesManager(strategies)
        assert len(manager.strategies) == 2
    
    def test_add_strategy(self):
        """Test adding strategies."""
        manager = RebalancingStrategiesManager()
        strategy = VolTargetRebalanceStrategy()
        manager.add_strategy(strategy)
        assert len(manager.strategies) == 1
        assert manager.strategies[0] is strategy
    
    def test_apply_all_no_strategies(self):
        """Test applying with no strategies returns original weights."""
        manager = RebalancingStrategiesManager()
        weights = pd.Series({"A": 0.5, "B": 0.5})
        result = manager.apply_all(weights)
        pd.testing.assert_series_equal(result, weights)
    
    def test_apply_all_combined_strategies(self):
        """Test applying multiple strategies in sequence."""
        vol_strategy = VolTargetRebalanceStrategy({
            "target": 0.05,  # Very low target to force leverage adjustment
            "lev_max": 2.0,
            "window": 3
        })
        guard_strategy = DrawdownGuardStrategy({
            "dd_threshold": 0.05,
            "guard_multiplier": 0.8
        })
        
        manager = RebalancingStrategiesManager([vol_strategy, guard_strategy])
        weights = pd.Series({"A": 0.5, "B": 0.5})
        
        # Setup some history for vol strategy with higher volatility
        returns = [0.02, -0.015, 0.018]  # Higher vol relative to low target
        for ret in returns:
            vol_strategy.apply(weights, current_returns=ret)
            guard_strategy.apply(weights, current_returns=ret)
        
        # Apply both strategies with another return
        result = manager.apply_all(
            weights,
            current_returns=0.02
        )
        
        # Should see some change due to volatility targeting
        leverage = result.sum() / weights.sum()
        assert leverage != 1.0  # Should be different from original
        
        # Should maintain same relative proportions
        expected_props = weights / weights.sum()
        actual_props = result / result.sum()
        pd.testing.assert_series_equal(actual_props, expected_props, rtol=1e-10)
    
    def test_reset_all_states(self):
        """Test resetting all strategy states."""
        vol_strategy = VolTargetRebalanceStrategy()
        guard_strategy = DrawdownGuardStrategy()
        
        manager = RebalancingStrategiesManager([vol_strategy, guard_strategy])
        weights = pd.Series({"A": 0.5, "B": 0.5})
        
        # Build up some state
        for ret in [0.01, -0.02, 0.005]:
            vol_strategy.apply(weights, current_returns=ret)
            guard_strategy.apply(weights, current_returns=ret)
        
        # Verify states exist
        assert len(vol_strategy._returns_buffer) > 0
        assert len(guard_strategy._cumulative_returns) > 0
        
        # Reset all
        manager.reset_all_states()
        
        # Verify states are reset
        assert len(vol_strategy._returns_buffer) == 0
        assert len(guard_strategy._cumulative_returns) == 0


class TestCreateRebalancingStrategy:
    """Tests for the strategy factory function."""
    
    def test_create_vol_target_strategy(self):
        """Test creating volatility targeting strategy."""
        params = {"target": 0.15, "window": 8}
        strategy = create_rebalancing_strategy("vol_target_rebalance", params)
        assert isinstance(strategy, VolTargetRebalanceStrategy)
        assert strategy.target == 0.15
        assert strategy.window == 8
    
    def test_create_drawdown_guard_strategy(self):
        """Test creating drawdown guard strategy."""
        params = {"dd_threshold": 0.12, "guard_multiplier": 0.6}
        strategy = create_rebalancing_strategy("drawdown_guard", params)
        assert isinstance(strategy, DrawdownGuardStrategy)
        assert strategy.dd_threshold == 0.12
        assert strategy.guard_multiplier == 0.6
    
    def test_create_strategy_no_params(self):
        """Test creating strategies without parameters."""
        vol_strategy = create_rebalancing_strategy("vol_target_rebalance")
        assert isinstance(vol_strategy, VolTargetRebalanceStrategy)
        
        guard_strategy = create_rebalancing_strategy("drawdown_guard")
        assert isinstance(guard_strategy, DrawdownGuardStrategy)
    
    def test_create_unknown_strategy(self):
        """Test error when creating unknown strategy."""
        with pytest.raises(ValueError, match="Unknown rebalancing strategy"):
            create_rebalancing_strategy("unknown_strategy")


class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""
    
    def test_vol_targeting_realistic_scenario(self):
        """Test vol targeting with realistic market scenarios."""
        strategy = VolTargetRebalanceStrategy({
            "target": 0.12,
            "window": 6,
            "lev_min": 0.5,
            "lev_max": 2.0
        })
        
        weights = pd.Series({"A": 0.4, "B": 0.35, "C": 0.25})
        
        # Simulate different market regimes
        
        # 1. Low volatility period
        low_vol_returns = [0.005, 0.002, 0.003, -0.001, 0.004, 0.001]
        for ret in low_vol_returns:
            result = strategy.apply(weights, current_returns=ret)
        
        # Should leverage up due to low volatility
        assert result.sum() > weights.sum()
        
        # 2. High volatility period (continue from previous state)
        high_vol_returns = [0.08, -0.06, 0.05, -0.04, 0.07, -0.05]
        for ret in high_vol_returns:
            result = strategy.apply(weights, current_returns=ret)
        
        # Should leverage down due to high volatility
        assert result.sum() < weights.sum()
    
    def test_drawdown_guard_realistic_scenario(self):
        """Test drawdown guard with realistic market crash scenario."""
        strategy = DrawdownGuardStrategy({
            "dd_window": 10,
            "dd_threshold": 0.15,
            "guard_multiplier": 0.3,
            "recover_threshold": 0.08
        })
        
        weights = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
        
        # Market crash scenario
        crash_returns = [
            0.01, 0.02, -0.08, -0.12, -0.05,  # Initial crash
            -0.03, -0.02, 0.01, -0.01,         # Continued weakness
        ]
        
        results = []
        for ret in crash_returns:
            result = strategy.apply(weights, current_returns=ret)
            results.append(result)
        
        # Should be in guard mode after crash
        assert strategy._in_guard_mode
        assert results[-1].sum() < weights.sum()
        
        # Recovery scenario
        recovery_returns = [0.05, 0.04, 0.06, 0.03, 0.02]
        for ret in recovery_returns:
            result = strategy.apply(weights, current_returns=ret)
        
        # Should exit guard mode after recovery
        assert not strategy._in_guard_mode
        pd.testing.assert_series_equal(result, weights)
    
    def test_combined_strategies_stress_test(self):
        """Test both strategies together under stress conditions."""
        vol_strategy = VolTargetRebalanceStrategy({
            "target": 0.10,
            "lev_min": 0.3,
            "lev_max": 3.0
        })
        
        guard_strategy = DrawdownGuardStrategy({
            "dd_threshold": 0.10,
            "guard_multiplier": 0.4
        })
        
        manager = RebalancingStrategiesManager([vol_strategy, guard_strategy])
        weights = pd.Series({"A": 0.5, "B": 0.5})
        
        # Volatile crash scenario
        stress_returns = [
            0.02, -0.15, 0.10, -0.08, -0.12,  # Volatile crash
            0.05, -0.03, 0.08, -0.02, 0.04    # Volatile recovery
        ]
        
        for ret in stress_returns:
            result = manager.apply_all(weights, current_returns=ret)
        
        # Both strategies should have had an effect
        assert not result.equals(weights)
        # Result should still be valid portfolio weights (all positive, maintain proportions)
        assert all(result >= 0)
        if result.sum() > 0:
            proportions = result / result.sum()
            original_proportions = weights / weights.sum()
            pd.testing.assert_series_equal(proportions, original_proportions, rtol=1e-10)