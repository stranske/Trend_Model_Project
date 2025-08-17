"""
Tests for rebalancing strategies.

Tests both trigger/no-trigger cases and hold vs full rebalance scenarios
as specified in the acceptance criteria.
"""

import pytest
import pandas as pd
import numpy as np

from trend_analysis.rebalancing import (
    DriftBandStrategy,
    PeriodicRebalanceStrategy,
    RebalancingEngine,
    RebalanceEvent,
    create_rebalancing_engine,
)


class TestDriftBandStrategy:
    """Test drift band rebalancing strategy."""
    
    def test_initialization(self):
        """Test strategy initialization with default parameters."""
        strategy = DriftBandStrategy()
        assert strategy.band_pct == 0.03
        assert strategy.min_trade == 0.005
        assert strategy.mode == "partial"
    
    def test_initialization_custom_params(self):
        """Test strategy initialization with custom parameters."""
        strategy = DriftBandStrategy(band_pct=0.05, min_trade=0.01, mode="full")
        assert strategy.band_pct == 0.05
        assert strategy.min_trade == 0.01
        assert strategy.mode == "full"
    
    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be 'partial' or 'full'"):
            DriftBandStrategy(mode="invalid")
    
    def test_no_trigger_small_drift(self):
        """Test no trigger when drift is below band."""
        strategy = DriftBandStrategy(band_pct=0.05, min_trade=0.01)
        
        current = pd.Series({"A": 0.52, "B": 0.48})  # drift of 0.02 from equal
        target = pd.Series({"A": 0.50, "B": 0.50})
        
        assert not strategy.should_trigger(current, target, period=1)
        
        result = strategy.apply_rebalance(current, target, period=1)
        assert not result.should_rebalance
        assert len(result.trades) == 0
        pd.testing.assert_series_equal(result.realized_weights, current)
    
    def test_no_trigger_below_min_trade(self):
        """Test no trigger when drift exceeds band but is below min_trade."""
        strategy = DriftBandStrategy(band_pct=0.02, min_trade=0.05)
        
        current = pd.Series({"A": 0.54, "B": 0.46})  # drift of 0.04, above band
        target = pd.Series({"A": 0.50, "B": 0.50})
        
        assert not strategy.should_trigger(current, target, period=1)
        
        result = strategy.apply_rebalance(current, target, period=1)
        assert not result.should_rebalance
        assert len(result.trades) == 0
    
    def test_trigger_drift_above_band_partial_mode(self):
        """Test trigger and partial rebalancing when drift exceeds band."""
        strategy = DriftBandStrategy(band_pct=0.03, min_trade=0.01, mode="partial")
        
        current = pd.Series({"A": 0.56, "B": 0.44})  # drift of 0.06 from equal
        target = pd.Series({"A": 0.50, "B": 0.50})
        
        assert strategy.should_trigger(current, target, period=1)
        
        result = strategy.apply_rebalance(current, target, period=1)
        assert result.should_rebalance
        assert len(result.trades) == 2
        
        # Should only trade back to band edge, not full target
        # A: 0.56 -> 0.53 (reduce by 0.03 to band edge)
        # B: 0.44 -> 0.47 (increase by 0.03 to band edge)
        expected_weights = pd.Series({"A": 0.53, "B": 0.47})
        pd.testing.assert_series_equal(result.realized_weights, expected_weights, check_exact=False, atol=1e-10)
        
        # Check trade details
        trade_a = next(t for t in result.trades if t.symbol == "A")
        trade_b = next(t for t in result.trades if t.symbol == "B")
        
        assert trade_a.trade_amount == pytest.approx(-0.03, abs=1e-10)  # sell A
        assert trade_b.trade_amount == pytest.approx(0.03, abs=1e-10)   # buy B
        assert "partial_rebalance" in trade_a.reason
        assert "partial_rebalance" in trade_b.reason
    
    def test_trigger_drift_above_band_full_mode(self):
        """Test trigger and full rebalancing when drift exceeds band."""
        strategy = DriftBandStrategy(band_pct=0.03, min_trade=0.01, mode="full")
        
        current = pd.Series({"A": 0.56, "B": 0.44})
        target = pd.Series({"A": 0.50, "B": 0.50})
        
        assert strategy.should_trigger(current, target, period=1)
        
        result = strategy.apply_rebalance(current, target, period=1)
        assert result.should_rebalance
        assert len(result.trades) == 2
        
        # Should fully rebalance to target weights
        pd.testing.assert_series_equal(result.realized_weights, target)
        
        # Check trade details
        trade_a = next(t for t in result.trades if t.symbol == "A")
        trade_b = next(t for t in result.trades if t.symbol == "B")
        
        assert trade_a.trade_amount == pytest.approx(-0.06, abs=1e-10)  # sell A
        assert trade_b.trade_amount == pytest.approx(0.06, abs=1e-10)   # buy B
        assert "full_rebalance" in trade_a.reason
        assert "full_rebalance" in trade_b.reason
    
    def test_misaligned_indices(self):
        """Test handling of misaligned current and target weight indices."""
        strategy = DriftBandStrategy(band_pct=0.03, min_trade=0.01, mode="full")
        
        current = pd.Series({"A": 0.6, "B": 0.4})
        target = pd.Series({"A": 0.4, "C": 0.6})  # B dropped, C added
        
        result = strategy.apply_rebalance(current, target, period=1)
        
        # Should handle missing indices with 0.0 weights
        expected_indices = {"A", "B", "C"}
        assert set(result.realized_weights.index) == expected_indices
        
        # A: 0.6 -> 0.4, B: 0.4 -> 0.0, C: 0.0 -> 0.6
        assert result.realized_weights.loc["A"] == pytest.approx(0.4)
        assert result.realized_weights.loc["B"] == pytest.approx(0.0)
        assert result.realized_weights.loc["C"] == pytest.approx(0.6)
    
    def test_three_asset_portfolio(self):
        """Test with three-asset portfolio."""
        strategy = DriftBandStrategy(band_pct=0.05, min_trade=0.02, mode="partial")
        
        current = pd.Series({"A": 0.40, "B": 0.35, "C": 0.25})
        target = pd.Series({"A": 0.33, "B": 0.33, "C": 0.34})
        
        # Drifts: A: +0.07, B: +0.02, C: -0.09
        # A and C exceed band of 0.05 and min_trade of 0.02
        
        result = strategy.apply_rebalance(current, target, period=1)
        assert result.should_rebalance
        
        # A should be reduced by (0.07 - 0.05) = 0.02
        # B should not change (drift 0.02 < band 0.05)
        # C should be increased by (0.09 - 0.05) = 0.04
        expected_weights = pd.Series({"A": 0.38, "B": 0.35, "C": 0.29})
        pd.testing.assert_series_equal(result.realized_weights, expected_weights, check_exact=False, atol=1e-10)


class TestPeriodicRebalanceStrategy:
    """Test periodic rebalancing strategy."""
    
    def test_initialization(self):
        """Test strategy initialization with default parameters."""
        strategy = PeriodicRebalanceStrategy()
        assert strategy.interval == 1
        assert strategy._last_rebalance_period is None
    
    def test_initialization_custom_interval(self):
        """Test strategy initialization with custom interval."""
        strategy = PeriodicRebalanceStrategy(interval=3)
        assert strategy.interval == 3
    
    def test_invalid_interval_raises_error(self):
        """Test that invalid interval raises ValueError."""
        with pytest.raises(ValueError, match="interval must be >= 1"):
            PeriodicRebalanceStrategy(interval=0)
    
    def test_trigger_first_period(self):
        """Test trigger on first period."""
        strategy = PeriodicRebalanceStrategy(interval=3)
        
        current = pd.Series({"A": 0.6, "B": 0.4})
        target = pd.Series({"A": 0.5, "B": 0.5})
        
        assert strategy.should_trigger(current, target, period=1)
        
        result = strategy.apply_rebalance(current, target, period=1)
        assert result.should_rebalance
        assert strategy._last_rebalance_period == 1
        
        # Should fully rebalance to target
        pd.testing.assert_series_equal(result.realized_weights, target)
        assert len(result.trades) == 2
        
        trade_a = next(t for t in result.trades if t.symbol == "A")
        trade_b = next(t for t in result.trades if t.symbol == "B")
        assert trade_a.trade_amount == pytest.approx(-0.1)
        assert trade_b.trade_amount == pytest.approx(0.1)
        assert "periodic_rebalance_interval_3" in trade_a.reason
    
    def test_no_trigger_before_interval(self):
        """Test no trigger before interval periods have passed."""
        strategy = PeriodicRebalanceStrategy(interval=3)
        
        current = pd.Series({"A": 0.6, "B": 0.4})
        target = pd.Series({"A": 0.5, "B": 0.5})
        
        # First rebalance at period 1
        strategy.apply_rebalance(current, target, period=1)
        assert strategy._last_rebalance_period == 1
        
        # Should not trigger at period 2 or 3
        assert not strategy.should_trigger(current, target, period=2)
        assert not strategy.should_trigger(current, target, period=3)
        
        result2 = strategy.apply_rebalance(current, target, period=2)
        assert not result2.should_rebalance
        assert len(result2.trades) == 0
        
        result3 = strategy.apply_rebalance(current, target, period=3)
        assert not result3.should_rebalance
        assert len(result3.trades) == 0
    
    def test_trigger_after_interval(self):
        """Test trigger after interval periods have passed."""
        strategy = PeriodicRebalanceStrategy(interval=3)
        
        current = pd.Series({"A": 0.6, "B": 0.4})
        target = pd.Series({"A": 0.5, "B": 0.5})
        
        # First rebalance at period 1
        strategy.apply_rebalance(current, target, period=1)
        
        # Should trigger at period 4 (1 + 3)
        assert strategy.should_trigger(current, target, period=4)
        
        result = strategy.apply_rebalance(current, target, period=4)
        assert result.should_rebalance
        assert strategy._last_rebalance_period == 4
    
    def test_interval_one_always_rebalances(self):
        """Test that interval=1 triggers every period."""
        strategy = PeriodicRebalanceStrategy(interval=1)
        
        current = pd.Series({"A": 0.6, "B": 0.4})
        target = pd.Series({"A": 0.5, "B": 0.5})
        
        # Should trigger on every period
        for period in [1, 2, 3, 4, 5]:
            assert strategy.should_trigger(current, target, period=period)
            result = strategy.apply_rebalance(current, target, period=period)
            assert result.should_rebalance
            assert strategy._last_rebalance_period == period
    
    def test_no_trades_when_weights_equal(self):
        """Test no trades when current equals target weights."""
        strategy = PeriodicRebalanceStrategy(interval=1)
        
        weights = pd.Series({"A": 0.5, "B": 0.5})
        
        result = strategy.apply_rebalance(weights, weights, period=1)
        assert result.should_rebalance  # Trigger is true, but no trades needed
        assert len(result.trades) == 0
        pd.testing.assert_series_equal(result.realized_weights, weights)


class TestRebalancingEngine:
    """Test the rebalancing engine that orchestrates multiple strategies."""
    
    def test_bayesian_only_mode(self):
        """Test that bayesian_only mode bypasses all strategies."""
        engine = RebalancingEngine(
            strategies=["drift_band", "periodic_rebalance"],
            bayesian_only=True
        )
        
        current = pd.Series({"A": 0.7, "B": 0.3})
        target = pd.Series({"A": 0.5, "B": 0.5})
        
        result = engine.apply_rebalancing(current, target, period=1)
        assert not result.should_rebalance
        assert len(result.trades) == 0
        pd.testing.assert_series_equal(result.realized_weights, target)
    
    def test_single_strategy_drift_band(self):
        """Test engine with single drift band strategy."""
        engine = RebalancingEngine(
            strategies=["drift_band"],
            params={"drift_band": {"band_pct": 0.05, "min_trade": 0.01, "mode": "full"}},
            bayesian_only=False
        )
        
        current = pd.Series({"A": 0.60, "B": 0.40})
        target = pd.Series({"A": 0.50, "B": 0.50})
        
        result = engine.apply_rebalancing(current, target, period=1)
        assert result.should_rebalance
        assert len(result.trades) == 2
        pd.testing.assert_series_equal(result.realized_weights, target)
    
    def test_single_strategy_periodic(self):
        """Test engine with single periodic strategy."""
        engine = RebalancingEngine(
            strategies=["periodic_rebalance"],
            params={"periodic_rebalance": {"interval": 2}},
            bayesian_only=False
        )
        
        current = pd.Series({"A": 0.60, "B": 0.40})
        target = pd.Series({"A": 0.50, "B": 0.50})
        
        # First period should trigger
        result1 = engine.apply_rebalancing(current, target, period=1)
        assert result1.should_rebalance
        
        # Second period should not trigger
        result2 = engine.apply_rebalancing(current, target, period=2)
        assert not result2.should_rebalance
        
        # Third period should trigger (1 + 2)
        result3 = engine.apply_rebalancing(current, target, period=3)
        assert result3.should_rebalance
    
    def test_multiple_strategies_sequence(self):
        """Test engine with multiple strategies applied in sequence."""
        engine = RebalancingEngine(
            strategies=["drift_band", "periodic_rebalance"],
            params={
                "drift_band": {"band_pct": 0.04, "min_trade": 0.01, "mode": "partial"},
                "periodic_rebalance": {"interval": 1}
            },
            bayesian_only=False
        )
        
        # Set up weights where drift band will trigger but only partially rebalance
        current = pd.Series({"A": 0.55, "B": 0.45})  # drift of 0.05 from equal
        target = pd.Series({"A": 0.50, "B": 0.50})
        
        result = engine.apply_rebalancing(current, target, period=1)
        assert result.should_rebalance
        
        # Both strategies should contribute trades
        assert len(result.trades) >= 2
        
        # Final result should be fully rebalanced (periodic comes after drift_band)
        pd.testing.assert_series_equal(result.realized_weights, target)
    
    def test_unknown_strategy_raises_error(self):
        """Test that unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy: unknown"):
            RebalancingEngine(
                strategies=["unknown"],
                bayesian_only=False
            )
    
    def test_empty_strategies_list(self):
        """Test engine with empty strategies list."""
        engine = RebalancingEngine(
            strategies=[],
            bayesian_only=False
        )
        
        current = pd.Series({"A": 0.60, "B": 0.40})
        target = pd.Series({"A": 0.50, "B": 0.50})
        
        result = engine.apply_rebalancing(current, target, period=1)
        assert not result.should_rebalance
        assert len(result.trades) == 0
        # When no strategies and bayesian_only=False, should return current weights
        pd.testing.assert_series_equal(result.realized_weights, current)


class TestCreateRebalancingEngine:
    """Test the factory function for creating rebalancing engines."""
    
    def test_create_with_full_config(self):
        """Test creating engine with full configuration."""
        config = {
            "rebalance": {
                "bayesian_only": False,
                "strategies": ["drift_band", "periodic_rebalance"],
                "params": {
                    "drift_band": {
                        "band_pct": 0.04,
                        "min_trade": 0.008,
                        "mode": "full"
                    },
                    "periodic_rebalance": {
                        "interval": 3
                    }
                }
            }
        }
        
        engine = create_rebalancing_engine(config)
        assert not engine.bayesian_only
        assert engine.strategies == ["drift_band", "periodic_rebalance"]
        assert len(engine._strategy_instances) == 2
        
        # Test that strategies have correct parameters
        drift_strategy = engine._strategy_instances[0]
        periodic_strategy = engine._strategy_instances[1]
        
        assert isinstance(drift_strategy, DriftBandStrategy)
        assert drift_strategy.band_pct == 0.04
        assert drift_strategy.min_trade == 0.008
        assert drift_strategy.mode == "full"
        
        assert isinstance(periodic_strategy, PeriodicRebalanceStrategy)
        assert periodic_strategy.interval == 3
    
    def test_create_with_defaults(self):
        """Test creating engine with minimal configuration (uses defaults)."""
        config = {}
        
        engine = create_rebalancing_engine(config)
        assert engine.bayesian_only  # Default is True
        assert engine.strategies == ["drift_band"]  # Default strategy
        assert len(engine._strategy_instances) == 0  # No instances when bayesian_only=True
    
    def test_create_with_partial_config(self):
        """Test creating engine with partial configuration."""
        config = {
            "rebalance": {
                "bayesian_only": False,
                "strategies": ["periodic_rebalance"]
            }
        }
        
        engine = create_rebalancing_engine(config)
        assert not engine.bayesian_only
        assert engine.strategies == ["periodic_rebalance"]
        assert len(engine._strategy_instances) == 1
        
        # Should use default parameters for periodic strategy
        periodic_strategy = engine._strategy_instances[0]
        assert isinstance(periodic_strategy, PeriodicRebalanceStrategy)
        assert periodic_strategy.interval == 1  # Default value


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_hold_vs_rebalance_scenario(self):
        """Test hold vs rebalance behavior based on triggers."""
        # Drift band strategy with tight band
        engine = RebalancingEngine(
            strategies=["drift_band"],
            params={"drift_band": {"band_pct": 0.02, "min_trade": 0.01, "mode": "partial"}},
            bayesian_only=False
        )
        
        target = pd.Series({"A": 0.5, "B": 0.5})
        
        # Scenario 1: Small drift - should hold
        small_drift = pd.Series({"A": 0.51, "B": 0.49})
        result1 = engine.apply_rebalancing(small_drift, target, period=1)
        assert not result1.should_rebalance  # Hold
        pd.testing.assert_series_equal(result1.realized_weights, small_drift)
        
        # Scenario 2: Large drift - should rebalance
        large_drift = pd.Series({"A": 0.54, "B": 0.46})
        result2 = engine.apply_rebalancing(large_drift, target, period=1)
        assert result2.should_rebalance  # Rebalance
        assert not result2.realized_weights.equals(large_drift)  # Weights changed
        assert len(result2.trades) > 0
    
    def test_compose_with_bayesian_weights(self):
        """Test composition behavior when bayesian_only=False."""
        # This is a conceptual test - in real usage, Bayesian weights would be 
        # the "target" weights passed to the rebalancing engine
        
        engine = RebalancingEngine(
            strategies=["periodic_rebalance"],
            params={"periodic_rebalance": {"interval": 1}},
            bayesian_only=False
        )
        
        # Current portfolio weights
        current = pd.Series({"A": 0.4, "B": 0.3, "C": 0.3})
        
        # "Target" weights as determined by Bayesian weighting
        bayesian_target = pd.Series({"A": 0.35, "B": 0.4, "C": 0.25})
        
        result = engine.apply_rebalancing(current, bayesian_target, period=1)
        assert result.should_rebalance
        
        # Final weights should match Bayesian target
        pd.testing.assert_series_equal(result.realized_weights, bayesian_target)
        
        # Trades should move from current to Bayesian target
        assert len(result.trades) == 3
        total_trade = sum(abs(t.trade_amount) for t in result.trades)
        expected_total = abs(current - bayesian_target).sum()
        assert total_trade == pytest.approx(expected_total)