"""
Rebalancing strategies for portfolio management.

This module implements various rebalancing strategies that can be applied to
portfolio weights during multi-period backtesting, including:

- vol_target_rebalance: Scale portfolio to maintain target volatility
- drawdown_guard: Reduce exposure during drawdown periods
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np
import pandas as pd


class RebalancingStrategy(ABC):
    """Base class for rebalancing strategies."""
    
    def __init__(self, params: Dict[str, Any] | None = None):
        self.params = params or {}
        
    @abstractmethod
    def apply(
        self,
        weights: pd.Series,
        returns_history: pd.DataFrame | None = None,
        **kwargs
    ) -> pd.Series:
        """
        Apply the rebalancing strategy to the given weights.
        
        Parameters
        ----------
        weights : pd.Series
            Current portfolio weights
        returns_history : pd.DataFrame, optional
            Historical returns for volatility/drawdown calculations
        **kwargs
            Additional strategy-specific parameters
            
        Returns
        -------
        pd.Series
            Adjusted portfolio weights
        """
        pass
        
    @abstractmethod
    def reset_state(self) -> None:
        """Reset strategy state for new backtest runs."""
        pass


class VolTargetRebalanceStrategy(RebalancingStrategy):
    """
    Volatility targeting strategy that scales portfolio weights to achieve
    a target volatility level within specified leverage bounds.
    
    Parameters
    ----------
    target : float, default 0.10
        Target annualized volatility (e.g., 0.10 for 10%)
    window : int, default 6
        Number of periods for volatility estimation
    estimator : str, default "simple"
        Volatility estimation method ("simple", "ewma")
    lev_min : float, default 0.5
        Minimum leverage multiplier
    lev_max : float, default 1.5
        Maximum leverage multiplier
    """
    
    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        self.target = float(self.params.get("target", 0.10))
        self.window = int(self.params.get("window", 6))
        self.estimator = str(self.params.get("estimator", "simple"))
        self.lev_min = float(self.params.get("lev_min", 0.5))
        self.lev_max = float(self.params.get("lev_max", 1.5))
        
        # State tracking
        self._returns_buffer: list[float] = []
        
    def apply(
        self,
        weights: pd.Series,
        returns_history: pd.DataFrame | None = None,
        current_returns: float | None = None,
        **kwargs
    ) -> pd.Series:
        """
        Apply volatility targeting to portfolio weights.
        
        Parameters
        ----------
        weights : pd.Series
            Current portfolio weights
        returns_history : pd.DataFrame, optional
            Historical returns DataFrame
        current_returns : float, optional
            Current period portfolio returns for state tracking
        
        Returns
        -------
        pd.Series
            Vol-adjusted portfolio weights
        """
        if weights.empty:
            return weights
            
        # Track returns for volatility estimation
        if current_returns is not None:
            self._returns_buffer.append(float(current_returns))
            if len(self._returns_buffer) > self.window:
                self._returns_buffer.pop(0)
        
        # Need at least 2 observations for volatility estimation
        if len(self._returns_buffer) < 2:
            return weights
            
        # Estimate current portfolio volatility
        returns_array = np.array(self._returns_buffer)
        
        if self.estimator == "ewma":
            # Simple EWMA with lambda=0.94 (RiskMetrics style)
            lambda_val = 0.94
            var_est = returns_array[-1] ** 2  # Start with most recent
            for i in range(len(returns_array) - 2, -1, -1):  # Work backwards
                var_est = lambda_val * var_est + (1 - lambda_val) * (returns_array[i] ** 2)
            current_vol = np.sqrt(var_est * 252)  # Annualized
        else:
            # Simple historical volatility
            current_vol = np.std(returns_array, ddof=1) * np.sqrt(252)
            
        # Avoid division by zero
        if current_vol <= 1e-10:
            return weights
            
        # Calculate leverage adjustment
        leverage = self.target / current_vol
        leverage = np.clip(leverage, self.lev_min, self.lev_max)
        
        # Apply leverage to weights
        adjusted_weights = weights * leverage
        
        # Ensure weights sum to leverage (not necessarily 1)
        return adjusted_weights
        
    def reset_state(self) -> None:
        """Reset volatility estimation buffer."""
        self._returns_buffer.clear()


class DrawdownGuardStrategy(RebalancingStrategy):
    """
    Drawdown guard strategy that reduces portfolio exposure when trailing
    drawdown exceeds a threshold and recovers when conditions improve.
    
    Parameters
    ----------
    dd_window : int, default 12
        Number of periods for drawdown calculation window
    dd_threshold : float, default 0.10
        Drawdown threshold to trigger guard (e.g., 0.10 for 10%)
    guard_multiplier : float, default 0.5
        Multiplier to apply when in guard mode (0.5 = 50% exposure)
    recover_threshold : float, default 0.05
        Maximum drawdown level to exit guard mode (e.g., 0.05 for 5%)
    """
    
    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        self.dd_window = int(self.params.get("dd_window", 12))
        self.dd_threshold = float(self.params.get("dd_threshold", 0.10))
        self.guard_multiplier = float(self.params.get("guard_multiplier", 0.5))
        self.recover_threshold = float(self.params.get("recover_threshold", 0.05))
        
        # State tracking
        self._cumulative_returns: list[float] = []
        self._in_guard_mode: bool = False
        
    def apply(
        self,
        weights: pd.Series,
        returns_history: pd.DataFrame | None = None,
        current_returns: float | None = None,
        **kwargs
    ) -> pd.Series:
        """
        Apply drawdown guard to portfolio weights.
        
        Parameters
        ----------
        weights : pd.Series
            Current portfolio weights
        returns_history : pd.DataFrame, optional
            Historical returns DataFrame
        current_returns : float, optional
            Current period portfolio returns for state tracking
            
        Returns
        -------
        pd.Series
            Guard-adjusted portfolio weights
        """
        if weights.empty:
            return weights
            
        # Track cumulative returns for drawdown calculation
        if current_returns is not None:
            if not self._cumulative_returns:
                self._cumulative_returns.append(1.0 + current_returns)
            else:
                prev_cum = self._cumulative_returns[-1]
                self._cumulative_returns.append(prev_cum * (1.0 + current_returns))
                
            # Keep only the required window
            if len(self._cumulative_returns) > self.dd_window:
                self._cumulative_returns.pop(0)
        
        # Need at least 2 observations for drawdown calculation
        if len(self._cumulative_returns) < 2:
            return weights
            
        # Calculate trailing drawdown
        cum_returns = np.array(self._cumulative_returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / peak
        current_drawdown = abs(drawdown[-1])  # Take absolute value
        
        # Update guard mode status
        if not self._in_guard_mode:
            # Enter guard mode if drawdown exceeds threshold
            if current_drawdown > self.dd_threshold:
                self._in_guard_mode = True
        else:
            # Exit guard mode if drawdown recovers below recovery threshold
            if current_drawdown <= self.recover_threshold:
                self._in_guard_mode = False
        
        # Apply guard multiplier if in guard mode
        if self._in_guard_mode:
            return weights * self.guard_multiplier
        else:
            return weights
            
    def reset_state(self) -> None:
        """Reset drawdown tracking state."""
        self._cumulative_returns.clear()
        self._in_guard_mode = False


class RebalancingStrategiesManager:
    """
    Manager class for applying multiple rebalancing strategies in sequence.
    """
    
    def __init__(self, strategies: list[RebalancingStrategy] | None = None):
        self.strategies = strategies or []
        
    def add_strategy(self, strategy: RebalancingStrategy) -> None:
        """Add a rebalancing strategy."""
        self.strategies.append(strategy)
        
    def apply_all(
        self,
        weights: pd.Series,
        **kwargs
    ) -> pd.Series:
        """
        Apply all strategies in sequence to the weights.
        
        Parameters
        ----------
        weights : pd.Series
            Initial portfolio weights
        **kwargs
            Parameters passed to each strategy's apply method
            
        Returns
        -------
        pd.Series
            Weights after applying all strategies
        """
        result_weights = weights.copy()
        
        for strategy in self.strategies:
            result_weights = strategy.apply(result_weights, **kwargs)
            
        return result_weights
        
    def reset_all_states(self) -> None:
        """Reset state for all strategies."""
        for strategy in self.strategies:
            strategy.reset_state()


def create_rebalancing_strategy(name: str, params: Dict[str, Any] | None = None) -> RebalancingStrategy:
    """
    Factory function to create rebalancing strategies by name.
    
    Parameters
    ----------
    name : str
        Strategy name ("vol_target_rebalance", "drawdown_guard")
    params : dict, optional
        Strategy parameters
        
    Returns
    -------
    RebalancingStrategy
        Created strategy instance
        
    Raises
    ------
    ValueError
        If strategy name is not recognized
    """
    if name == "vol_target_rebalance":
        return VolTargetRebalanceStrategy(params)
    elif name == "drawdown_guard":
        return DrawdownGuardStrategy(params)
    else:
        raise ValueError(f"Unknown rebalancing strategy: {name}")