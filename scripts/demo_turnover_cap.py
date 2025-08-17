#!/usr/bin/env python3
"""
Demonstration of turnover_cap rebalancing strategy.

This script shows how the turnover_cap strategy works with different scenarios.
Run this script to see the turnover cap rebalancing strategy in action.

Usage:
    python demo_turnover_cap.py

Requirements:
    - Virtual environment must be activated: source .venv/bin/activate
    - PYTHONPATH must include src: PYTHONPATH="./src" python demo_turnover_cap.py
"""

import pandas as pd
import numpy as np
from trend_analysis.rebalancing import TurnoverCapStrategy

def demo_basic_turnover_cap():
    """Demonstrate basic turnover cap functionality."""
    print("=== Basic Turnover Cap Demo ===")
    
    # Setup strategy with 20% max turnover and 10 bps costs
    strategy = TurnoverCapStrategy({
        "max_turnover": 0.2,
        "cost_bps": 10,
        "priority": "largest_gap"
    })
    
    # Current portfolio weights
    current = pd.Series([0.5, 0.3, 0.2], index=['Stock_A', 'Stock_B', 'Stock_C'])
    
    # Target portfolio weights (major rebalancing needed)
    target = pd.Series([0.2, 0.6, 0.2], index=['Stock_A', 'Stock_B', 'Stock_C'])
    
    print(f"Current weights:\n{current}")
    print(f"Target weights:\n{target}")
    
    desired_turnover = (target - current).abs().sum()
    print(f"Desired turnover: {desired_turnover:.1%}")
    
    # Apply strategy
    new_weights, cost = strategy.apply(current, target)
    
    actual_turnover = (new_weights - current).abs().sum()
    print(f"Actual turnover: {actual_turnover:.1%} (capped at {strategy.max_turnover:.1%})")
    print(f"Transaction cost: ${cost*10000:.2f} per $10,000 invested")
    print(f"Final weights:\n{new_weights}")
    print()

def demo_priority_mechanisms():
    """Demonstrate different priority mechanisms."""
    print("=== Priority Mechanisms Demo ===")
    
    current = pd.Series([0.4, 0.3, 0.3], index=['Stock_A', 'Stock_B', 'Stock_C'])
    target = pd.Series([0.1, 0.5, 0.4], index=['Stock_A', 'Stock_B', 'Stock_C'])
    scores = pd.Series([1.0, 2.5, 0.8], index=['Stock_A', 'Stock_B', 'Stock_C'])  # B has highest score
    
    print(f"Current: {current.to_dict()}")
    print(f"Target:  {target.to_dict()}")
    print(f"Scores:  {scores.to_dict()}")
    
    # Test largest_gap priority
    strategy1 = TurnoverCapStrategy({"max_turnover": 0.2, "priority": "largest_gap", "cost_bps": 0})
    weights1, _ = strategy1.apply(current, target)
    print(f"\nLargest Gap Priority: {weights1.to_dict()}")
    
    # Test best_score_delta priority  
    strategy2 = TurnoverCapStrategy({"max_turnover": 0.2, "priority": "best_score_delta", "cost_bps": 0})
    weights2, _ = strategy2.apply(current, target, scores=scores)
    print(f"Score Delta Priority: {weights2.to_dict()}")
    
    print()

def demo_integration_example():
    """Demonstrate integration with multi-period engine."""
    print("=== Multi-Period Integration Demo ===")
    
    from trend_analysis.multi_period.engine import run_schedule
    from trend_analysis.selector import RankSelector
    from trend_analysis.weighting import EqualWeight
    
    # Create sample score frames for 3 periods
    dates = ["2020-01", "2020-02", "2020-03"]
    assets = ["Fund_A", "Fund_B", "Fund_C", "Fund_D"]
    
    score_frames = {}
    np.random.seed(42)  # For reproducible results
    for i, date in enumerate(dates):
        # Different performance rankings each period
        scores = np.random.normal(0.1, 0.05, len(assets)) + i * 0.01
        score_frames[date] = pd.DataFrame({
            "Sharpe": scores,
            "CAGR": np.random.normal(0.08, 0.02, len(assets))
        }, index=assets)
    
    # Setup selection and weighting
    selector = RankSelector(top_n=2, rank_column="Sharpe")
    weighting = EqualWeight()
    
    # Run with turnover cap
    rebalance_strategies = ["turnover_cap"]
    rebalance_params = {
        "turnover_cap": {
            "max_turnover": 0.15,  # 15% max turnover per period
            "cost_bps": 25,        # 25 bps transaction costs
            "priority": "best_score_delta"
        }
    }
    
    portfolio = run_schedule(
        score_frames,
        selector,
        weighting,
        rebalance_strategies=rebalance_strategies,
        rebalance_params=rebalance_params
    )
    
    print("Portfolio evolution with 15% turnover cap:")
    for date, weights in portfolio.history.items():
        print(f"{date}: {weights.to_dict()}")
    
    print(f"Total transaction costs: ${portfolio.total_rebalance_costs*10000:.2f} per $10,000")
    print()

def main():
    """Run all demonstrations."""
    print("Turnover Cap Rebalancing Strategy Demonstration")
    print("=" * 50)
    print()
    
    demo_basic_turnover_cap()
    demo_priority_mechanisms()
    demo_integration_example()
    
    print("All demonstrations completed successfully!")

if __name__ == "__main__":
    main()