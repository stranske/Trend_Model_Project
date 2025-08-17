#!/usr/bin/env python3
"""
Demo script for rebalancing strategies.
"""

import pandas as pd
import numpy as np
from trend_analysis.rebalancing import (
    VolTargetRebalanceStrategy,
    DrawdownGuardStrategy,
    RebalancingStrategiesManager
)


def create_sample_returns(n_periods=20):
    """Create sample returns data for demonstration."""
    np.random.seed(42)
    # Create different market regimes
    low_vol_returns = np.random.normal(0.008, 0.015, n_periods // 4)  # Low vol
    high_vol_returns = np.random.normal(-0.005, 0.040, n_periods // 2)  # High vol crash
    recovery_returns = np.random.normal(0.012, 0.020, n_periods // 4)  # Recovery
    
    return np.concatenate([low_vol_returns, high_vol_returns, recovery_returns])


def demo_vol_targeting():
    """Demonstrate volatility targeting strategy."""
    print("=" * 50)
    print("VOLATILITY TARGETING STRATEGY DEMO")
    print("=" * 50)
    
    strategy = VolTargetRebalanceStrategy({
        "target": 0.12,  # 12% target volatility
        "window": 6,
        "lev_min": 0.3,
        "lev_max": 3.0
    })
    
    base_weights = pd.Series({"Fund_A": 0.4, "Fund_B": 0.35, "Fund_C": 0.25})
    returns = create_sample_returns(15)
    
    print(f"Base portfolio weights: {dict(base_weights)}")
    print(f"Target volatility: {strategy.target:.1%}")
    print(f"Leverage bounds: {strategy.lev_min:.1f}x to {strategy.lev_max:.1f}x")
    print()
    
    results = []
    for i, ret in enumerate(returns):
        adjusted_weights = strategy.apply(base_weights, current_returns=ret)
        leverage = adjusted_weights.sum() / base_weights.sum()
        
        results.append({
            'Period': i + 1,
            'Return': f"{ret:.1%}",
            'Leverage': f"{leverage:.2f}x",
            'Total_Exposure': f"{adjusted_weights.sum():.1%}"
        })
        
        if i < 5 or i >= len(returns) - 3:  # Show first 5 and last 3
            print(f"Period {i+1:2d}: Return={ret:+.1%}, "
                  f"Leverage={leverage:.2f}x, "
                  f"Exposure={adjusted_weights.sum():.1%}")
    
    print("...")
    return results


def demo_drawdown_guard():
    """Demonstrate drawdown guard strategy."""
    print("\n" + "=" * 50)
    print("DRAWDOWN GUARD STRATEGY DEMO")
    print("=" * 50)
    
    strategy = DrawdownGuardStrategy({
        "dd_window": 8,
        "dd_threshold": 0.12,  # 12% drawdown trigger
        "guard_multiplier": 0.4,  # 40% exposure when guarding
        "recover_threshold": 0.06  # 6% to exit guard
    })
    
    base_weights = pd.Series({"Fund_A": 0.5, "Fund_B": 0.5})
    
    # Create a specific scenario with drawdown
    scenario_returns = [
        0.02, 0.01, -0.08, -0.05, -0.03,  # Drawdown phase
        -0.01, 0.04, 0.06, 0.03, 0.02      # Recovery phase
    ]
    
    print(f"Base portfolio weights: {dict(base_weights)}")
    print(f"Drawdown threshold: {strategy.dd_threshold:.1%}")
    print(f"Guard multiplier: {strategy.guard_multiplier:.1%}")
    print(f"Recovery threshold: {strategy.recover_threshold:.1%}")
    print()
    
    cumulative_return = 1.0
    for i, ret in enumerate(scenario_returns):
        cumulative_return *= (1 + ret)
        adjusted_weights = strategy.apply(base_weights, current_returns=ret)
        guard_status = "GUARD" if strategy._in_guard_mode else "NORMAL"
        exposure = adjusted_weights.sum()
        
        print(f"Period {i+1:2d}: Return={ret:+.1%}, "
              f"Cum.Return={cumulative_return-1:+.1%}, "
              f"Status={guard_status:6s}, "
              f"Exposure={exposure:.1%}")


def demo_combined_strategies():
    """Demonstrate both strategies working together."""
    print("\n" + "=" * 50)
    print("COMBINED STRATEGIES DEMO")
    print("=" * 50)
    
    vol_strategy = VolTargetRebalanceStrategy({
        "target": 0.10,
        "window": 5,
        "lev_max": 2.0
    })
    
    guard_strategy = DrawdownGuardStrategy({
        "dd_window": 6,
        "dd_threshold": 0.10,
        "guard_multiplier": 0.6
    })
    
    manager = RebalancingStrategiesManager([vol_strategy, guard_strategy])
    base_weights = pd.Series({"Fund_A": 0.6, "Fund_B": 0.4})
    
    # Volatile scenario with both high vol and drawdown
    volatile_returns = [
        0.01, 0.05, -0.12, 0.08, -0.06,
        -0.04, 0.03, 0.07, -0.02, 0.04
    ]
    
    print(f"Base portfolio weights: {dict(base_weights)}")
    print("Applying both vol targeting (10% target) and drawdown guard (10% threshold)")
    print()
    
    for i, ret in enumerate(volatile_returns):
        # Apply both strategies
        adjusted_weights = manager.apply_all(base_weights, current_returns=ret)
        
        vol_leverage = adjusted_weights.sum() / base_weights.sum()
        guard_status = "GUARD" if guard_strategy._in_guard_mode else "NORMAL"
        
        print(f"Period {i+1:2d}: Return={ret:+.1%}, "
              f"Leverage={vol_leverage:.2f}x, "
              f"Guard={guard_status:6s}, "
              f"Final Exposure={adjusted_weights.sum():.1%}")


if __name__ == "__main__":
    print("Rebalancing Strategies Demo")
    print("This demonstrates vol_target_rebalance and drawdown_guard strategies\n")
    
    demo_vol_targeting()
    demo_drawdown_guard() 
    demo_combined_strategies()
    
    print("\n" + "=" * 50)
    print("Demo completed! Check config/rebalancing_demo.yml for configuration example.")