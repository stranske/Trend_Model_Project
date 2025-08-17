"""
Non-Bayesian rebalancing strategies for portfolio management.

This module implements various rebalancing strategies that determine when and how
to adjust portfolio weights based on drift, time intervals, and other criteria.
These strategies can be composed with Bayesian weighting methods when configured
with bayesian_only=false.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List
import pandas as pd


@dataclass
class RebalanceEvent:
    """Represents a single rebalancing trade."""
    symbol: str
    current_weight: float
    target_weight: float
    trade_amount: float  # positive = buy, negative = sell
    reason: str


@dataclass
class RebalanceResult:
    """Result of applying a rebalancing strategy."""
    realized_weights: pd.Series  # Final weights after rebalancing
    trades: List[RebalanceEvent]  # List of trades executed
    should_rebalance: bool  # Whether any rebalancing was triggered


class BaseRebalanceStrategy(ABC):
    """Abstract base class for rebalancing strategies."""
    
    def __init__(self, **params: Any) -> None:
        self.params = params
        self._last_rebalance_period: int | None = None
    
    @abstractmethod
    def should_trigger(
        self, 
        current_weights: pd.Series,
        target_weights: pd.Series,
        period: int
    ) -> bool:
        """Determine if rebalancing should be triggered."""
        pass
    
    @abstractmethod
    def apply_rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        period: int
    ) -> RebalanceResult:
        """Apply the rebalancing strategy and return trades and realized weights."""
        pass


class DriftBandStrategy(BaseRebalanceStrategy):
    """
    Rebalance when portfolio weights drift outside specified bands.
    
    Parameters
    ----------
    band_pct : float, default 0.03
        Maximum allowed drift as absolute percentage (e.g., 0.03 = 3%).
    min_trade : float, default 0.005  
        Minimum trade size as absolute weight (e.g., 0.005 = 0.5%).
    mode : str, default "partial"
        Rebalancing mode:
        - "partial": Only trade the excess drift above the band
        - "full": Fully rebalance back to target weights
    """
    
    def __init__(
        self,
        band_pct: float = 0.03,
        min_trade: float = 0.005,
        mode: str = "partial",
        **params: Any
    ) -> None:
        super().__init__(**params)
        self.band_pct = float(band_pct)
        self.min_trade = float(min_trade)
        self.mode = str(mode).lower()
        if self.mode not in ["partial", "full"]:
            raise ValueError(f"mode must be 'partial' or 'full', got '{self.mode}'")
    
    def should_trigger(
        self, 
        current_weights: pd.Series,
        target_weights: pd.Series,
        period: int
    ) -> bool:
        """Check if any weight has drifted outside the band."""
        # Align indices
        aligned_current, aligned_target = current_weights.align(target_weights, fill_value=0.0)
        
        # Calculate drift as absolute difference
        drifts = (aligned_current - aligned_target).abs()
        
        # Check if any drift exceeds band and minimum trade size
        max_drift = drifts.max()
        return max_drift > self.band_pct and max_drift > self.min_trade
    
    def apply_rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        period: int
    ) -> RebalanceResult:
        """Apply drift band rebalancing."""
        # Align indices  
        aligned_current, aligned_target = current_weights.align(target_weights, fill_value=0.0)
        
        trades = []
        
        if not self.should_trigger(current_weights, target_weights, period):
            return RebalanceResult(
                realized_weights=aligned_current,
                trades=[],
                should_rebalance=False
            )
        
        if self.mode == "full":
            # Full rebalancing: trade to exact target weights
            realized_weights = aligned_target.copy()
            for symbol in aligned_current.index:
                current_w = aligned_current.loc[symbol]
                target_w = aligned_target.loc[symbol]
                trade_amount = target_w - current_w
                
                if abs(trade_amount) >= self.min_trade:
                    trades.append(RebalanceEvent(
                        symbol=symbol,
                        current_weight=current_w,
                        target_weight=target_w,
                        trade_amount=trade_amount,
                        reason=f"full_rebalance_drift_band"
                    ))
        else:
            # Partial rebalancing: only trade excess drift
            drifts = aligned_current - aligned_target
            realized_weights = aligned_current.copy()
            
            for symbol in aligned_current.index:
                drift = drifts.loc[symbol]
                abs_drift = abs(drift)
                
                if abs_drift > self.band_pct and abs_drift >= self.min_trade:
                    # Trade back to the band edge
                    if drift > 0:
                        # Overweight: sell excess above band
                        trade_amount = -(abs_drift - self.band_pct)
                    else:
                        # Underweight: buy to reduce deficit to band
                        trade_amount = abs_drift - self.band_pct
                    
                    if abs(trade_amount) >= self.min_trade:
                        new_weight = aligned_current.loc[symbol] + trade_amount
                        realized_weights.loc[symbol] = new_weight
                        
                        trades.append(RebalanceEvent(
                            symbol=symbol,
                            current_weight=aligned_current.loc[symbol],
                            target_weight=aligned_target.loc[symbol],
                            trade_amount=trade_amount,
                            reason=f"partial_rebalance_drift_band"
                        ))
        
        return RebalanceResult(
            realized_weights=realized_weights,
            trades=trades,
            should_rebalance=True
        )


class PeriodicRebalanceStrategy(BaseRebalanceStrategy):
    """
    Rebalance at regular time intervals.
    
    Parameters
    ----------
    interval : int, default 1
        Number of periods between rebalances.
    """
    
    def __init__(self, interval: int = 1, **params: Any) -> None:
        super().__init__(**params)
        self.interval = int(interval)
        if self.interval < 1:
            raise ValueError(f"interval must be >= 1, got {self.interval}")
    
    def should_trigger(
        self, 
        current_weights: pd.Series,
        target_weights: pd.Series,
        period: int
    ) -> bool:
        """Check if enough periods have passed since last rebalance."""
        if self._last_rebalance_period is None:
            # First period, trigger rebalance
            return True
        
        periods_since_last = period - self._last_rebalance_period
        return periods_since_last >= self.interval
    
    def apply_rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        period: int
    ) -> RebalanceResult:
        """Apply periodic rebalancing."""
        # Align indices
        aligned_current, aligned_target = current_weights.align(target_weights, fill_value=0.0)
        
        if not self.should_trigger(current_weights, target_weights, period):
            return RebalanceResult(
                realized_weights=aligned_current,
                trades=[],
                should_rebalance=False
            )
        
        # Update last rebalance period
        self._last_rebalance_period = period
        
        # Full rebalancing to target weights
        trades = []
        for symbol in aligned_current.index:
            current_w = aligned_current.loc[symbol]
            target_w = aligned_target.loc[symbol]
            trade_amount = target_w - current_w
            
            if abs(trade_amount) > 1e-8:  # Avoid tiny floating point trades
                trades.append(RebalanceEvent(
                    symbol=symbol,
                    current_weight=current_w,
                    target_weight=target_w,
                    trade_amount=trade_amount,
                    reason=f"periodic_rebalance_interval_{self.interval}"
                ))
        
        return RebalanceResult(
            realized_weights=aligned_target.copy(),
            trades=trades,
            should_rebalance=True
        )


class RebalancingEngine:
    """
    Orchestrates multiple rebalancing strategies.
    
    This engine can compose non-Bayesian strategies with Bayesian weighting
    when bayesian_only=false.
    """
    
    def __init__(
        self,
        strategies: List[str] | None = None,
        params: Dict[str, Dict[str, Any]] | None = None,
        bayesian_only: bool = True
    ) -> None:
        """
        Initialize the rebalancing engine.
        
        Parameters
        ----------
        strategies : list of str, optional
            List of strategy names to apply in order. Default is ["drift_band"].
        params : dict, optional
            Parameters for each strategy, keyed by strategy name.
        bayesian_only : bool, default True
            If True, skip non-Bayesian strategies and rely only on Bayesian weighting.
        """
        self.strategies = strategies if strategies is not None else ["drift_band"]
        self.params = params or {}
        self.bayesian_only = bayesian_only
        self._strategy_instances: List[BaseRebalanceStrategy] = []
        
        if not self.bayesian_only:
            self._initialize_strategies()
    
    def _initialize_strategies(self) -> None:
        """Initialize strategy instances based on configuration."""
        strategy_map = {
            "drift_band": DriftBandStrategy,
            "periodic_rebalance": PeriodicRebalanceStrategy,
        }
        
        for strategy_name in self.strategies:
            if strategy_name not in strategy_map:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            
            strategy_class = strategy_map[strategy_name]
            strategy_params = self.params.get(strategy_name, {})
            instance = strategy_class(**strategy_params)
            self._strategy_instances.append(instance)
    
    def apply_rebalancing(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        period: int
    ) -> RebalanceResult:
        """
        Apply all configured rebalancing strategies in sequence.
        
        Returns
        -------
        RebalanceResult
            Combined result with final weights and all trades.
        """
        if self.bayesian_only or not self._strategy_instances:
            # Just return current weights - no rebalancing when bayesian_only or no strategies
            # When bayesian_only=True, the target weights are already the final Bayesian weights
            return RebalanceResult(
                realized_weights=target_weights.copy() if self.bayesian_only else current_weights.copy(),
                trades=[],
                should_rebalance=False
            )
        
        working_weights = current_weights.copy()
        all_trades: List[RebalanceEvent] = []
        any_rebalance = False
        
        # Apply strategies in sequence
        for strategy in self._strategy_instances:
            result = strategy.apply_rebalance(working_weights, target_weights, period)
            working_weights = result.realized_weights
            all_trades.extend(result.trades)
            any_rebalance = any_rebalance or result.should_rebalance
        
        return RebalanceResult(
            realized_weights=working_weights,
            trades=all_trades,
            should_rebalance=any_rebalance
        )


def create_rebalancing_engine(config: Dict[str, Any]) -> RebalancingEngine:
    """
    Create a rebalancing engine from configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with rebalancing settings.
        Expected structure:
        {
            "rebalance": {
                "bayesian_only": bool,
                "strategies": list of str,
                "params": {
                    "drift_band": {"band_pct": float, "min_trade": float, "mode": str},
                    "periodic_rebalance": {"interval": int}
                }
            }
        }
    
    Returns
    -------
    RebalancingEngine
        Configured rebalancing engine.
    """
    rebalance_config = config.get("rebalance", {})
    
    return RebalancingEngine(
        strategies=rebalance_config.get("strategies", ["drift_band"]),
        params=rebalance_config.get("params", {}),
        bayesian_only=rebalance_config.get("bayesian_only", True)
    )
