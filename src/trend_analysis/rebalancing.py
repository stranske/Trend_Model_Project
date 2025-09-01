"""Portfolio rebalancing strategies implementation.

This module provides various rebalancing strategies that control how target weights
are realized into actual trades and positions, including turnover constraints and
transaction cost modeling.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

# Small epsilon value for turnover comparisons to handle numerical precision
TURNOVER_EPSILON = 1e-10


class RebalancingStrategy(ABC):
    """Base class for rebalancing strategies."""

    def __init__(self, params: Dict[str, Any] | None = None):
        self.params = params or {}

    @abstractmethod
    def apply(
        self, current_weights: pd.Series, target_weights: pd.Series, **kwargs
    ) -> Tuple[pd.Series, float]:
        """Apply the rebalancing strategy.

        Parameters
        ----------
        current_weights : pd.Series
            Current portfolio weights
        target_weights : pd.Series
            Target portfolio weights
        **kwargs
            Additional context (scores, prices, etc.)

        Returns
        -------
        tuple[pd.Series, float]
            New weights after rebalancing and total cost incurred
        """
        pass


class TurnoverCapStrategy(RebalancingStrategy):
    """Turnover cap rebalancing strategy.

    Limits the total turnover (sum of absolute trades) per rebalancing period
    and applies optional transaction costs. Prioritizes trades by either
    largest gap or best score delta.
    """

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        self.max_turnover = float(self.params.get("max_turnover", 0.2))
        self.cost_bps = int(self.params.get("cost_bps", 10))
        self.priority = str(self.params.get("priority", "largest_gap"))

    def apply(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        scores: Optional[pd.Series] = None,
        **kwargs,
    ) -> Tuple[pd.Series, float]:
        """Apply turnover cap with trade prioritization and cost modeling.

        Parameters
        ----------
        current_weights : pd.Series
            Current portfolio weights
        target_weights : pd.Series
            Target portfolio weights
        scores : pd.Series, optional
            Asset scores for priority calculation (when priority='best_score_delta')
        **kwargs
            Additional context

        Returns
        -------
        tuple[pd.Series, float]
            New weights after turnover cap and total transaction cost
        """
        # Align indices
        all_assets = current_weights.index.union(target_weights.index)
        current = current_weights.reindex(all_assets, fill_value=0.0)
        target = target_weights.reindex(all_assets, fill_value=0.0)

        # Calculate desired trades
        trades = target - current
        total_desired_turnover = trades.abs().sum()

        # If within turnover limit, execute all trades
        if total_desired_turnover <= self.max_turnover:
            actual_turnover = total_desired_turnover
            new_weights = target.copy()
        else:
            # Need to scale trades to respect turnover cap
            new_weights, actual_turnover = self._apply_turnover_cap(
                current, target, trades, scores
            )

        # Apply transaction costs
        cost = self._calculate_cost(actual_turnover)

        return new_weights, cost

    def _apply_turnover_cap(
        self,
        current: pd.Series,
        target: pd.Series,
        trades: pd.Series,
        scores: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, float]:
        """Apply turnover cap with prioritized trade allocation."""

        # Calculate trade priorities
        priorities = self._calculate_priorities(current, target, trades, scores)

        # Sort trades by priority (highest first)
        trade_items = [
            (asset, trade, priority)
            for asset, trade, priority in zip(
                trades.index, trades.values, priorities.values
            )
        ]
        trade_items.sort(key=lambda x: x[2], reverse=True)

        # Allocate turnover budget by priority
        remaining_turnover = self.max_turnover
        executed_trades = pd.Series(0.0, index=trades.index)

        for asset, desired_trade, priority in trade_items:
            if (
                remaining_turnover <= TURNOVER_EPSILON
            ):  # Check if remaining turnover is negligible
                break

            # Scale trade to fit remaining budget
            trade_size = abs(desired_trade)
            if (
                trade_size <= remaining_turnover + TURNOVER_EPSILON
            ):  # Allow for numerical precision tolerance
                # Execute full trade
                executed_trades[asset] = desired_trade
                remaining_turnover -= trade_size
            else:
                # Execute partial trade within remaining budget
                if desired_trade != 0:
                    scale_factor = remaining_turnover / trade_size
                    executed_trades[asset] = desired_trade * scale_factor
                    remaining_turnover = 0

        # Apply executed trades
        new_weights = current + executed_trades

        # Ensure weights are non-negative
        new_weights = new_weights.clip(lower=0.0)

        actual_turnover = executed_trades.abs().sum()

        return new_weights, actual_turnover

    def _calculate_priorities(
        self,
        current: pd.Series,
        target: pd.Series,
        trades: pd.Series,
        scores: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Calculate trade priorities based on configured priority method."""

        if self.priority == "largest_gap":
            # Prioritize by absolute size of the trade
            priorities = trades.abs()

        elif self.priority == "best_score_delta":
            if scores is None:
                # Fall back to largest_gap if no scores provided
                priorities = trades.abs()
            else:
                # Prioritize by score-weighted trade benefit
                # For positions we're increasing, use positive score
                # For positions we're decreasing, use negative score (higher priority for dropping low-scored assets)
                scores_aligned = scores.reindex(trades.index, fill_value=0.0)
                priorities = trades * scores_aligned
                # Take absolute value to ensure highest absolute priority wins
                priorities = priorities.abs()
        else:
            # Default to largest_gap
            priorities = trades.abs()

        return priorities

    def _calculate_cost(self, turnover: float) -> float:
        """Calculate transaction cost based on turnover and cost basis points."""
        return turnover * (self.cost_bps / 10000.0)


class PeriodicRebalanceStrategy(RebalancingStrategy):
    """Periodic rebalance strategy - rebalance every N periods."""

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        self.interval = int(self.params.get("interval", 1))
        self._period_count = 0

    def apply(
        self, current_weights: pd.Series, target_weights: pd.Series, **kwargs
    ) -> Tuple[pd.Series, float]:
        """Apply periodic rebalancing."""
        self._period_count += 1

        if self._period_count >= self.interval:
            # Time to rebalance
            self._period_count = 0
            all_assets = current_weights.index.union(target_weights.index)
            new_weights = target_weights.reindex(all_assets, fill_value=0.0)
            # No transaction costs in basic implementation
            cost = 0.0
        else:
            # Keep current weights
            new_weights = current_weights.copy()
            cost = 0.0

        return new_weights, cost


class DriftBandStrategy(RebalancingStrategy):
    """Drift band rebalancing strategy - rebalance when weights drift beyond bands."""

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        self.band_pct = float(self.params.get("band_pct", 0.03))
        self.min_trade = float(self.params.get("min_trade", 0.005))
        self.mode = str(self.params.get("mode", "partial"))

    def apply(
        self, current_weights: pd.Series, target_weights: pd.Series, **kwargs
    ) -> Tuple[pd.Series, float]:
        """Apply drift band rebalancing."""
        # Align indices
        all_assets = current_weights.index.union(target_weights.index)
        current = current_weights.reindex(all_assets, fill_value=0.0)
        target = target_weights.reindex(all_assets, fill_value=0.0)

        # Calculate drift from target
        drift = (current - target).abs()
        needs_rebalance = drift > self.band_pct

        if needs_rebalance.any():
            if self.mode == "full":
                # Full rebalance when any asset drifts
                new_weights = target.copy()
            else:  # partial
                # Only rebalance assets that have drifted
                new_weights = current.copy()
                trades = target - current
                # Only execute trades above minimum size
                significant_trades = trades.abs() > self.min_trade
                rebalance_assets = needs_rebalance & significant_trades
                new_weights[rebalance_assets] = target[rebalance_assets]
        else:
            new_weights = current.copy()

        # No transaction costs in basic implementation
        cost = 0.0
        return new_weights, cost


class VolTargetRebalanceStrategy(RebalancingStrategy):
    """Volatility target rebalancing strategy - scale positions to maintain target volatility."""

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        self.target_vol = float(self.params.get("target", 0.10))
        self.lev_min = float(self.params.get("lev_min", 0.5))
        self.lev_max = float(self.params.get("lev_max", 1.5))
        self.window = int(self.params.get("window", 6))

    def apply(
        self, current_weights: pd.Series, target_weights: pd.Series, **kwargs
    ) -> Tuple[pd.Series, float]:
        """Apply volatility targeting by scaling positions."""
        # Align indices
        all_assets = current_weights.index.union(target_weights.index)
        current = current_weights.reindex(all_assets, fill_value=0.0)
        target = target_weights.reindex(all_assets, fill_value=0.0)

        # Get equity curve from kwargs (maintained by rb_state)
        equity_curve = kwargs.get("equity_curve", [])

        if len(equity_curve) >= self.window + 1:
            # Compute realized volatility from past returns
            returns = pd.Series(
                np.diff(equity_curve[-(self.window + 1) :])
                / equity_curve[-(self.window + 1) : -1]
            )
            realized_vol = float(returns.std(ddof=0)) * np.sqrt(12)  # Annualized

            if realized_vol > 0:
                # Calculate leverage factor to achieve target vol
                lev_factor = np.clip(
                    self.target_vol / realized_vol, self.lev_min, self.lev_max
                )
                new_weights = target * lev_factor
            else:
                new_weights = target.copy()
        else:
            # Not enough history, use target weights as-is
            new_weights = target.copy()

        # No transaction costs in basic implementation
        cost = 0.0
        return new_weights, cost


class DrawdownGuardStrategy(RebalancingStrategy):
    """Drawdown guard strategy - reduce exposure during drawdown periods."""

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        self.dd_window = int(self.params.get("dd_window", 12))
        self.dd_threshold = float(self.params.get("dd_threshold", 0.10))
        self.guard_multiplier = float(self.params.get("guard_multiplier", 0.5))
        self.recover_threshold = float(self.params.get("recover_threshold", 0.05))
        self._guard_on = False

    def apply(
        self, current_weights: pd.Series, target_weights: pd.Series, **kwargs
    ) -> Tuple[pd.Series, float]:
        """Apply drawdown protection by scaling positions."""
        # Align indices
        all_assets = current_weights.index.union(target_weights.index)
        current = current_weights.reindex(all_assets, fill_value=0.0)
        target = target_weights.reindex(all_assets, fill_value=0.0)

        # Get equity curve and guard state from kwargs
        equity_curve = kwargs.get("equity_curve", [])
        if "rb_state" in kwargs and "guard_on" in kwargs["rb_state"]:
            self._guard_on = kwargs["rb_state"]["guard_on"]
        # else, keep self._guard_on as is
        drawdown = 0.0
        if len(equity_curve) >= 1:
            # Calculate drawdown over the specified window
            window_data = (
                equity_curve[-self.dd_window :]
                if len(equity_curve) >= self.dd_window
                else equity_curve
            )
            peak = max(window_data)
            current_value = window_data[-1]
            if peak > 0:
                drawdown = (current_value / peak) - 1.0

        # Update guard state
        if not self._guard_on and drawdown <= -self.dd_threshold:
            self._guard_on = True
        elif self._guard_on and drawdown >= -self.recover_threshold:
            self._guard_on = False

        # Store updated guard state back to kwargs (for state tracking)
        if "rb_state" in kwargs:
            kwargs["rb_state"]["guard_on"] = self._guard_on

        # Apply guard multiplier if guard is on
        if self._guard_on:
            new_weights = target * self.guard_multiplier
        else:
            new_weights = target.copy()

        # No transaction costs in basic implementation
        cost = 0.0
        return new_weights, cost


# Registry of available strategies
REBALANCING_STRATEGIES = {
    "turnover_cap": TurnoverCapStrategy,
    "periodic_rebalance": PeriodicRebalanceStrategy,
    "drift_band": DriftBandStrategy,
    "vol_target_rebalance": VolTargetRebalanceStrategy,
    "drawdown_guard": DrawdownGuardStrategy,
}


def create_rebalancing_strategy(
    name: str, params: Dict[str, Any] | None = None
) -> RebalancingStrategy:
    """Create a rebalancing strategy by name."""
    if name not in REBALANCING_STRATEGIES:
        raise ValueError(
            f"Unknown rebalancing strategy: {name}. Available: {list(REBALANCING_STRATEGIES.keys())}"
        )

    strategy_cls = REBALANCING_STRATEGIES[name]
    return strategy_cls(params)


def apply_rebalancing_strategies(
    strategies: List[str],
    strategy_params: Dict[str, Dict[str, Any]],
    current_weights: pd.Series,
    target_weights: pd.Series,
    **kwargs,
) -> Tuple[pd.Series, float]:
    """Apply multiple rebalancing strategies in sequence.

    Parameters
    ----------
    strategies : list[str]
        List of strategy names to apply in order
    strategy_params : dict
        Parameters for each strategy
    current_weights : pd.Series
        Current portfolio weights
    target_weights : pd.Series
        Target portfolio weights
    **kwargs
        Additional context for strategies

    Returns
    -------
    tuple[pd.Series, float]
        Final weights after all strategies and total cost
    """
    weights = current_weights.copy()
    total_cost = 0.0

    for strategy_name in strategies:
        params = strategy_params.get(strategy_name, {})
        strategy = create_rebalancing_strategy(strategy_name, params)
        weights, cost = strategy.apply(weights, target_weights, **kwargs)
        total_cost += cost

    return weights, total_cost


__all__ = [
    "RebalancingStrategy",
    "TurnoverCapStrategy",
    "PeriodicRebalanceStrategy",
    "DriftBandStrategy",
    "VolTargetRebalanceStrategy",
    "DrawdownGuardStrategy",
    "REBALANCING_STRATEGIES",
    "create_rebalancing_strategy",
    "apply_rebalancing_strategies",
]
