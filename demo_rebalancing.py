#!/usr/bin/env python3
"""
Demonstration script for rebalancing strategies.
Shows both trigger/no-trigger and hold vs rebalance scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from trend_analysis.rebalancing import DriftBandStrategy, PeriodicRebalanceStrategy, RebalancingEngine


def demo_drift_band():
    """Demonstrate drift band rebalancing strategy."""
    print("=== Drift Band Strategy Demo ===")
    
    strategy = DriftBandStrategy(band_pct=0.03, min_trade=0.01, mode="partial")
    target = pd.Series({"A": 0.50, "B": 0.50})
    
    print("Target weights:", target.to_dict())
    print("Band: ±3%, Min trade: 1%")
    print()
    
    # Scenario 1: Small drift - should hold
    small_drift = pd.Series({"A": 0.52, "B": 0.48})
    result1 = strategy.apply_rebalance(small_drift, target, period=1)
    print("Scenario 1 - Small drift (2%): HOLD")
    print(f"  Current: {small_drift.to_dict()}")
    print(f"  Should rebalance: {result1.should_rebalance}")
    print(f"  Trades: {len(result1.trades)}")
    print()
    
    # Scenario 2: Large drift - should rebalance  
    large_drift = pd.Series({"A": 0.56, "B": 0.44})
    result2 = strategy.apply_rebalance(large_drift, target, period=1)
    print("Scenario 2 - Large drift (6%): REBALANCE")
    print(f"  Current: {large_drift.to_dict()}")
    print(f"  Should rebalance: {result2.should_rebalance}")
    print(f"  Final weights: {result2.realized_weights.to_dict()}")
    print(f"  Trades: {len(result2.trades)}")
    for trade in result2.trades:
        print(f"    {trade.symbol}: {trade.trade_amount:+.3f} ({trade.reason})")
    print()


def demo_periodic_rebalance():
    """Demonstrate periodic rebalancing strategy."""
    print("=== Periodic Rebalance Strategy Demo ===")
    
    strategy = PeriodicRebalanceStrategy(interval=3)
    target = pd.Series({"A": 0.40, "B": 0.35, "C": 0.25})
    current = pd.Series({"A": 0.50, "B": 0.30, "C": 0.20})
    
    print("Target weights:", target.to_dict()) 
    print("Rebalance interval: 3 periods")
    print()
    
    for period in [1, 2, 3, 4]:
        result = strategy.apply_rebalance(current, target, period=period)
        action = "REBALANCE" if result.should_rebalance else "HOLD"
        print(f"Period {period}: {action}")
        print(f"  Should rebalance: {result.should_rebalance}")
        print(f"  Trades: {len(result.trades)}")
        if result.trades:
            print(f"  Final weights: {result.realized_weights.to_dict()}")
        print()


def demo_composition():
    """Demonstrate composition of multiple strategies."""
    print("=== Strategy Composition Demo ===")
    
    engine = RebalancingEngine(
        strategies=["drift_band", "periodic_rebalance"],
        params={
            "drift_band": {"band_pct": 0.04, "min_trade": 0.01, "mode": "partial"},
            "periodic_rebalance": {"interval": 1}
        },
        bayesian_only=False
    )
    
    current = pd.Series({"A": 0.55, "B": 0.45})  # 5% drift from equal
    target = pd.Series({"A": 0.50, "B": 0.50})
    
    print("Strategies: drift_band (4% band) + periodic_rebalance (interval=1)")
    print(f"Current: {current.to_dict()}")
    print(f"Target: {target.to_dict()}")
    print()
    
    result = engine.apply_rebalancing(current, target, period=1)
    print(f"Final result: {result.realized_weights.to_dict()}")
    print(f"Total trades: {len(result.trades)}")
    print("Trade sequence:")
    for i, trade in enumerate(result.trades, 1):
        print(f"  {i}. {trade.symbol}: {trade.trade_amount:+.3f} ({trade.reason})")
    print()


def demo_bayesian_composition():
    """Demonstrate composition with Bayesian weighting."""
    print("=== Bayesian Integration Demo ===")
    
    # Bayesian-only mode
    engine1 = RebalancingEngine(bayesian_only=True)
    current = pd.Series({"A": 0.40, "B": 0.30, "C": 0.30})
    bayesian_target = pd.Series({"A": 0.35, "B": 0.40, "C": 0.25})  # From Bayesian weighting
    
    result1 = engine1.apply_rebalancing(current, bayesian_target, period=1)
    print("Bayesian-only mode:")
    print(f"  Current: {current.to_dict()}")
    print(f"  Bayesian target: {bayesian_target.to_dict()}")
    print(f"  Final: {result1.realized_weights.to_dict()}")
    print(f"  Trades: {len(result1.trades)} (Bayesian handles everything)")
    print()
    
    # Composed mode - non-Bayesian strategies applied to Bayesian targets
    engine2 = RebalancingEngine(
        strategies=["drift_band"],
        params={"drift_band": {"band_pct": 0.02, "mode": "full"}},
        bayesian_only=False
    )
    
    result2 = engine2.apply_rebalancing(current, bayesian_target, period=1)
    print("Composed mode (Bayesian + drift_band):")
    print(f"  Current: {current.to_dict()}")
    print(f"  Bayesian target: {bayesian_target.to_dict()}")
    print(f"  Final: {result2.realized_weights.to_dict()}")
    print(f"  Trades: {len(result2.trades)}")
    for trade in result2.trades:
        print(f"    {trade.symbol}: {trade.trade_amount:+.3f}")
    print()


if __name__ == "__main__":
    print("Rebalancing Strategies Demonstration")
    print("===================================")
    print()
    
    demo_drift_band()
    demo_periodic_rebalance()
    demo_composition()
    demo_bayesian_composition()
    
    print("Demo complete! ✓")