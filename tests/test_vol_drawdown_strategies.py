"""Tests for volatility targeting and drawdown guard rebalancing strategies."""

import pandas as pd
import pytest

from trend_analysis.rebalancing import (
    DrawdownGuardStrategy,
    VolTargetRebalanceStrategy,
    create_rebalancing_strategy,
)


class TestVolTargetRebalanceStrategy:
    """Test volatility targeting rebalancing strategy."""

    def test_no_equity_curve_passthrough(self):
        """Test behavior when no equity curve is provided."""
        strategy = VolTargetRebalanceStrategy({"target": 0.10, "window": 6})
        current = pd.Series([0.3, 0.4, 0.3], index=["A", "B", "C"])
        target = pd.Series([0.4, 0.3, 0.3], index=["A", "B", "C"])

        result, cost = strategy.apply(current, target)

        # Should pass through target weights unchanged
        pd.testing.assert_series_equal(result, target)
        assert cost == 0.0

    def test_insufficient_history_passthrough(self):
        """Test behavior when equity curve is too short."""
        strategy = VolTargetRebalanceStrategy({"target": 0.10, "window": 6})
        current = pd.Series([0.5, 0.5], index=["A", "B"])
        target = pd.Series([0.6, 0.4], index=["A", "B"])

        # Only 5 points, need window+1=7
        equity_curve = [1.0, 1.02, 0.98, 1.01, 0.99]
        result, cost = strategy.apply(current, target, equity_curve=equity_curve)

        # Should pass through target weights unchanged
        pd.testing.assert_series_equal(result, target)
        assert cost == 0.0

    def test_vol_scaling_within_bounds(self):
        """Test volatility scaling respects leverage bounds."""
        strategy = VolTargetRebalanceStrategy(
            {"target": 0.10, "window": 6, "lev_min": 0.5, "lev_max": 1.5}
        )
        current = pd.Series([0.5, 0.5], index=["A", "B"])
        target = pd.Series([0.5, 0.5], index=["A", "B"])

        # Create high volatility equity curve to trigger lev_min
        equity_curve = [1.0]
        for ret in [0.08, -0.06, 0.07, -0.05, 0.06, -0.04, 0.05]:
            equity_curve.append(equity_curve[-1] * (1.0 + ret))

        result, cost = strategy.apply(current, target, equity_curve=equity_curve)

        # Should be scaled down due to high vol, but within bounds
        gross = result.sum()
        assert 0.5 <= gross <= 1.5
        assert cost == 0.0

    def test_low_vol_scaling_up(self):
        """Test scaling up when realized volatility is low."""
        strategy = VolTargetRebalanceStrategy(
            {"target": 0.10, "window": 6, "lev_min": 0.5, "lev_max": 2.0}
        )
        current = pd.Series([0.5, 0.5], index=["A", "B"])
        target = pd.Series([0.5, 0.5], index=["A", "B"])

        # Create low volatility equity curve
        equity_curve = [1.0]
        for ret in [0.005, 0.003, 0.004, 0.002, 0.001, 0.003, 0.002]:
            equity_curve.append(equity_curve[-1] * (1.0 + ret))

        result, cost = strategy.apply(current, target, equity_curve=equity_curve)

        # Should be scaled up due to low vol
        assert result.sum() > target.sum()
        assert result.sum() <= 2.0  # Respects lev_max
        assert cost == 0.0

    def test_zero_vol_edge_case(self):
        """Test handling of zero volatility edge case."""
        strategy = VolTargetRebalanceStrategy({"target": 0.10, "window": 3})
        current = pd.Series([0.5, 0.5], index=["A", "B"])
        target = pd.Series([0.6, 0.4], index=["A", "B"])

        # Flat equity curve (zero volatility)
        equity_curve = [1.0, 1.0, 1.0, 1.0]
        result, cost = strategy.apply(current, target, equity_curve=equity_curve)

        # Should pass through target when vol is zero
        pd.testing.assert_series_equal(result, target)
        assert cost == 0.0


class TestDrawdownGuardStrategy:
    """Test drawdown guard rebalancing strategy."""

    def test_no_equity_curve_passthrough(self):
        """Test behavior when no equity curve is provided."""
        strategy = DrawdownGuardStrategy({"dd_threshold": 0.10})
        current = pd.Series([0.4, 0.6], index=["A", "B"])
        target = pd.Series([0.5, 0.5], index=["A", "B"])

        result, cost = strategy.apply(current, target)

        # Should pass through target weights unchanged
        pd.testing.assert_series_equal(result, target)
        assert cost == 0.0

    def test_no_drawdown_passthrough(self):
        """Test behavior when no significant drawdown exists."""
        strategy = DrawdownGuardStrategy({"dd_threshold": 0.10, "guard_multiplier": 0.5})
        current = pd.Series([0.5, 0.5], index=["A", "B"])
        target = pd.Series([0.6, 0.4], index=["A", "B"])

        # Positive trending equity curve
        equity_curve = [1.0, 1.02, 1.05, 1.03, 1.08, 1.10]
        rb_state = {}
        result, cost = strategy.apply(current, target, equity_curve=equity_curve, rb_state=rb_state)

        # Should pass through target weights unchanged
        pd.testing.assert_series_equal(result, target)
        assert not rb_state.get("guard_on", False)
        assert cost == 0.0

    def test_drawdown_triggers_guard(self):
        """Test that significant drawdown triggers the guard."""
        strategy = DrawdownGuardStrategy(
            {"dd_threshold": 0.10, "guard_multiplier": 0.5, "dd_window": 5}
        )
        current = pd.Series([0.6, 0.4], index=["A", "B"])
        target = pd.Series([0.6, 0.4], index=["A", "B"])

        # Equity curve with >10% drawdown
        equity_curve = [1.0, 1.02, 0.98, 0.95, 0.90, 0.88]
        rb_state = {}
        result, cost = strategy.apply(current, target, equity_curve=equity_curve, rb_state=rb_state)

        # Should apply guard multiplier
        expected = target * 0.5
        pd.testing.assert_series_equal(result, expected)
        assert rb_state.get("guard_on", False) is True
        assert cost == 0.0

    def test_guard_recovery_turns_off(self):
        """Test that guard turns off when drawdown recovers."""
        strategy = DrawdownGuardStrategy(
            {"dd_threshold": 0.10, "guard_multiplier": 0.5, "recover_threshold": 0.05}
        )
        current = pd.Series([0.5, 0.5], index=["A", "B"])
        target = pd.Series([0.5, 0.5], index=["A", "B"])

        # Equity curve that has recovered (only -3% drawdown)
        equity_curve = [1.0, 1.05, 1.10, 1.08, 1.07, 1.07]
        rb_state = {"guard_on": True}
        result, cost = strategy.apply(current, target, equity_curve=equity_curve, rb_state=rb_state)

        # Should turn off guard and pass through target
        pd.testing.assert_series_equal(result, target)
        assert rb_state.get("guard_on", False) is False
        assert cost == 0.0

    def test_guard_persistence_during_dd(self):
        """Test that guard stays on during continued drawdown."""
        strategy = DrawdownGuardStrategy(
            {"dd_threshold": 0.10, "guard_multiplier": 0.3, "recover_threshold": 0.05}
        )
        current = pd.Series([0.4, 0.6], index=["A", "B"])
        target = pd.Series([0.4, 0.6], index=["A", "B"])

        # Start with guard on and continued drawdown
        equity_curve = [1.0, 1.05, 0.95, 0.90, 0.85, 0.82]  # Worsening DD
        rb_state = {"guard_on": True}
        result, cost = strategy.apply(current, target, equity_curve=equity_curve, rb_state=rb_state)

        # Should keep guard on and apply multiplier
        expected = target * 0.3
        pd.testing.assert_series_equal(result, expected)
        assert rb_state.get("guard_on", False) is True
        assert cost == 0.0

    def test_create_strategy_integration(self):
        """Test integration with strategy factory."""
        params = {
            "dd_threshold": 0.15,
            "guard_multiplier": 0.4,
            "recover_threshold": 0.08,
            "dd_window": 8,
        }
        strategy = create_rebalancing_strategy("drawdown_guard", params)

        assert isinstance(strategy, DrawdownGuardStrategy)
        # Smoke apply to ensure params accepted
        current = pd.Series([0.5, 0.5], index=["A", "B"])
        target = pd.Series([0.6, 0.4], index=["A", "B"])
        _, _ = strategy.apply(current, target, equity_curve=[1.0, 0.9, 0.95])

    def test_empty_equity_curve_edge_case(self):
        """Test handling of empty equity curve."""
        strategy = DrawdownGuardStrategy({"dd_threshold": 0.10})
        current = pd.Series([0.5, 0.5], index=["A", "B"])
        target = pd.Series([0.6, 0.4], index=["A", "B"])

        equity_curve = []
        rb_state = {}
        result, cost = strategy.apply(current, target, equity_curve=equity_curve, rb_state=rb_state)

        # Should pass through target when no curve provided
        pd.testing.assert_series_equal(result, target)
        assert not rb_state.get("guard_on", False)
        assert cost == 0.0


class TestStrategyIntegration:
    """Test integration of strategies with the factory system."""

    def test_vol_target_factory_creation(self):
        """Test creation via factory."""
        params = {"target": 0.08, "lev_min": 0.3, "lev_max": 2.5, "window": 12}
        strategy = create_rebalancing_strategy("vol_target_rebalance", params)

        assert isinstance(strategy, VolTargetRebalanceStrategy)
        current = pd.Series([0.5, 0.5], index=["A", "B"])
        target = pd.Series([0.6, 0.4], index=["A", "B"])
        _, _ = strategy.apply(current, target, equity_curve=[1.0, 1.01, 0.99])

    def test_strategy_registry_includes_new_strategies(self):
        """Test that new strategies are in the registry."""
        from trend_analysis.rebalancing import rebalancer_registry

        keys = set(rebalancer_registry.available())
        assert {"vol_target_rebalance", "drawdown_guard"}.issubset(keys)

    def test_unknown_strategy_raises_error(self):
        """Test that unknown strategy names raise appropriate error."""
        with pytest.raises(ValueError, match="Unknown plugin"):
            create_rebalancing_strategy("nonexistent_strategy")
