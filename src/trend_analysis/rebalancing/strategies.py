"""Portfolio rebalancing strategies implementation.

This module provides various rebalancing strategies that control how
target weights are realised into actual trades and positions, including
turnover constraints and transaction cost modelling.  Strategies are
exposed via a simple plugin registry so they can be selected by name in
configuration files.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..cash_policy import CashPolicy
from ..plugins import Rebalancer, create_rebalancer, rebalancer_registry

# Backwards compatibility name
RebalancingStrategy = Rebalancer

# Small epsilon value for turnover comparisons to handle numerical precision
TURNOVER_EPSILON = 1e-10


def _apply_cash_policy(weights: pd.Series, cash_policy: CashPolicy | None) -> pd.Series:
    if cash_policy is None:
        return weights

    updated = weights.copy()
    total = float(updated.sum()) if not updated.empty else 0.0

    if cash_policy.explicit_cash:
        if "CASH" in updated.index:
            non_cash = updated.drop(labels=["CASH"])
            updated.loc["CASH"] = 1.0 - float(non_cash.sum())
        else:
            updated.loc["CASH"] = 1.0 - total
        total = float(updated.sum())

    if cash_policy.normalize_weights and not np.isclose(total, 0.0):
        updated = updated / total

    return updated


@rebalancer_registry.register("turnover_cap")
class TurnoverCapStrategy(Rebalancer):
    """Turnover cap rebalancing strategy.

    Limits the total turnover (sum of absolute trades) per rebalancing
    period and applies optional transaction costs. Prioritizes trades by
    either largest gap or best score delta.
    Cash handling is controlled by ``cash_policy``: set ``normalize_weights``
    to force weights to sum to one, or ``explicit_cash`` to add a CASH line
    for any unallocated mass (negative values indicate financing).
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
        cash_policy: CashPolicy | None = None,
        scores: Optional[pd.Series] = None,
        **kwargs: Any,
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
            new_weights = _apply_cash_policy(target.copy(), cash_policy)
        else:
            # Need to scale trades to respect turnover cap
            new_weights, actual_turnover = self._apply_turnover_cap(
                current, target, trades, scores, cash_policy=cash_policy
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
        cash_policy: CashPolicy | None = None,
    ) -> Tuple[pd.Series, float]:
        """Apply turnover cap with prioritized trade allocation."""

        # Calculate trade priorities and order trades descending
        priorities = self._calculate_priorities(current, target, trades, scores)
        priorities = priorities.fillna(-np.inf)
        order = priorities.sort_values(ascending=False, kind="mergesort").index

        ordered_trades = trades.reindex(order)
        trade_values = ordered_trades.to_numpy(dtype=float, copy=False)
        abs_trades = np.abs(trade_values)

        cumsum_turnover = np.cumsum(abs_trades)
        full_mask = cumsum_turnover <= (self.max_turnover + TURNOVER_EPSILON)

        executed_ordered = np.zeros_like(trade_values)
        executed_ordered[full_mask] = trade_values[full_mask]

        remaining_turnover = max(0.0, self.max_turnover - abs_trades[full_mask].sum())

        if remaining_turnover > TURNOVER_EPSILON:
            # Execute partial trade for the next highest-priority asset
            remaining_indices = np.flatnonzero(~full_mask & (abs_trades > 0))
            if remaining_indices.size:
                idx = int(remaining_indices[0])
                executed_ordered[idx] = np.sign(trade_values[idx]) * min(
                    abs_trades[idx], remaining_turnover
                )
                remaining_turnover = 0.0

        executed_trades = pd.Series(0.0, index=trades.index)
        executed_trades.loc[order] = executed_ordered

        # Apply executed trades
        new_weights = current + executed_trades

        # Ensure weights are non-negative
        new_weights = new_weights.clip(lower=0.0)
        new_weights = _apply_cash_policy(new_weights, cash_policy)

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
                # For positions we're decreasing, use negative score
                # (higher priority for dropping low-scored assets)
                scores_aligned = scores.reindex(trades.index, fill_value=0.0)
                priorities = trades * scores_aligned
                # Take absolute value to ensure highest absolute priority wins
                priorities = priorities.abs()
        else:
            # Default to largest_gap
            priorities = trades.abs()

        return priorities

    def _calculate_cost(self, turnover: float) -> float:
        """Calculate transaction cost based on turnover and cost basis
        points."""
        return turnover * (self.cost_bps / 10000.0)


@rebalancer_registry.register("periodic_rebalance")
class PeriodicRebalanceStrategy(Rebalancer):
    """Periodic rebalance strategy - rebalance every N periods.

    Cash handling is controlled by ``cash_policy``: set ``normalize_weights``
    to force weights to sum to one, or ``explicit_cash`` to add a CASH line
    for any unallocated mass (negative values indicate financing).
    """

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        self.interval = int(self.params.get("interval", 1))
        self._period_count = 0

    def apply(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        cash_policy: CashPolicy | None = None,
        **kwargs: Any,
    ) -> Tuple[pd.Series, float]:
        """Apply periodic rebalancing."""
        self._period_count += 1

        if self._period_count >= self.interval:
            # Time to rebalance
            self._period_count = 0
            all_assets = current_weights.index.union(target_weights.index)
            new_weights = target_weights.reindex(all_assets, fill_value=0.0)
            new_weights = _apply_cash_policy(new_weights, cash_policy)
            # No transaction costs in basic implementation
            cost = 0.0
        else:
            # Keep current weights
            new_weights = _apply_cash_policy(current_weights.copy(), cash_policy)
            cost = 0.0

        return new_weights, cost


@rebalancer_registry.register("drift_band")
class DriftBandStrategy(Rebalancer):
    """Drift band rebalancing strategy - rebalance when weights drift beyond bands.

    Cash handling is controlled by ``cash_policy``: set ``normalize_weights``
    to force weights to sum to one, or ``explicit_cash`` to add a CASH line
    for any unallocated mass (negative values indicate financing).
    """

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        self.band_pct = float(self.params.get("band_pct", 0.03))
        self.min_trade = float(self.params.get("min_trade", 0.005))
        self.mode = str(self.params.get("mode", "partial"))

    def apply(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        cash_policy: CashPolicy | None = None,
        **kwargs: Any,
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

        new_weights = _apply_cash_policy(new_weights, cash_policy)

        # No transaction costs in basic implementation
        cost = 0.0
        return new_weights, cost


@rebalancer_registry.register("vol_target_rebalance")
class VolTargetRebalanceStrategy(Rebalancer):
    """Scale weights to hit a target volatility based on recent equity
    curve.

    Cash handling is controlled by ``cash_policy``: set ``normalize_weights``
    to force weights to sum to one, or ``explicit_cash`` to add a CASH line
    for any unallocated mass (negative values indicate financing).
    """

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.target = float(self.params.get("target", 0.10))
        self.window = int(self.params.get("window", 6))
        self.lev_min = float(self.params.get("lev_min", 0.5))
        self.lev_max = float(self.params.get("lev_max", 1.5))
        self.financing_spread_bps = float(self.params.get("financing_spread_bps", 0.0))

    def apply(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        cash_policy: CashPolicy | None = None,
        **kwargs: Any,
    ) -> Tuple[pd.Series, float]:
        ec: List[float] = list(kwargs.get("equity_curve", []))
        lev = 1.0
        if len(ec) >= self.window + 1:
            rets = pd.Series(np.diff(ec[-(self.window + 1) :]) / ec[-(self.window + 1) : -1])
            vol = float(rets.std(ddof=0)) * np.sqrt(12)
            if vol > 0:
                lev = float(np.clip(self.target / vol, self.lev_min, self.lev_max))
        # Scale target weights by leverage; pass through target when no equity curve
        scaled = target_weights * lev
        scaled = _apply_cash_policy(scaled, cash_policy)
        financing_cost = 0.0
        if lev > 1.0 and self.financing_spread_bps > 0.0:
            financing_cost = (lev - 1.0) * (self.financing_spread_bps / 10000.0)
        return scaled, financing_cost


@rebalancer_registry.register("drawdown_guard")
class DrawdownGuardStrategy(Rebalancer):
    """Reduce exposure when portfolio experiences a drawdown.

    Cash handling is controlled by ``cash_policy``: set ``normalize_weights``
    to force weights to sum to one, or ``explicit_cash`` to add a CASH line
    for any unallocated mass (negative values indicate financing).
    """

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.dd_window = int(self.params.get("dd_window", 12))
        self.dd_threshold = float(self.params.get("dd_threshold", 0.10))
        self.guard_multiplier = float(self.params.get("guard_multiplier", 0.5))
        self.recover_threshold = float(self.params.get("recover_threshold", 0.05))

    def apply(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        cash_policy: CashPolicy | None = None,
        **kwargs: Any,
    ) -> Tuple[pd.Series, float]:
        # Prefer explicit rb_state dict if provided, else fallback to generic state, else a local dict
        rb_state_obj = kwargs.get("rb_state", kwargs.get("state"))
        rb_state: Dict[str, Any] = rb_state_obj if isinstance(rb_state_obj, dict) else {}
        # Equity curve can be passed either directly or via state
        ec_in: Any = kwargs.get("equity_curve", rb_state.get("equity_curve", []))
        ec: List[float] = list(ec_in)
        guard_on = bool(rb_state.get("guard_on", False))
        dd = 0.0
        if ec:
            sub = ec[-self.dd_window :] if len(ec) >= self.dd_window else ec
            peak = max(sub)
            cur = sub[-1]
            if peak > 0:
                dd = (cur / peak) - 1.0
        if (not guard_on and dd <= -self.dd_threshold) or (
            guard_on and dd <= -self.recover_threshold
        ):
            guard_on = True
        elif guard_on and dd >= -self.recover_threshold:
            guard_on = False
        # Persist state back
        rb_state["guard_on"] = guard_on
        # Apply guard by scaling target weights; otherwise pass through target weights
        scaled = target_weights * self.guard_multiplier if guard_on else target_weights
        scaled = _apply_cash_policy(scaled, cash_policy)
        return scaled, 0.0


# Registry of available strategies
def create_rebalancing_strategy(name: str, params: Dict[str, Any] | None = None) -> Rebalancer:
    """Create a rebalancing strategy by name using the plugin registry."""
    return create_rebalancer(name, params)


def apply_rebalancing_strategies(
    strategies: List[str],
    strategy_params: Dict[str, Dict[str, Any]],
    current_weights: pd.Series,
    target_weights: pd.Series,
    cash_policy: CashPolicy | None = None,
    **kwargs: Any,
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
    cash_policy : CashPolicy, optional
        Policy for handling implicit cash and normalization
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
        weights, cost = strategy.apply(weights, target_weights, cash_policy=cash_policy, **kwargs)
        total_cost += cost

    return weights, total_cost


__all__ = [
    "CashPolicy",
    "RebalancingStrategy",
    "TurnoverCapStrategy",
    "PeriodicRebalanceStrategy",
    "DriftBandStrategy",
    "VolTargetRebalanceStrategy",
    "DrawdownGuardStrategy",
    "create_rebalancing_strategy",
    "apply_rebalancing_strategies",
    "rebalancer_registry",
]
