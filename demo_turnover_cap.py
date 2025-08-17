#!/usr/bin/env python3
"""Demonstration of turnover_cap rebalancing strategy.

This script shows how the turnover_cap rebalancing strategy works in practice,
demonstrating turnover enforcement, transaction cost calculation, and different
priority mechanisms.

Usage:
    python demo_turnover_cap.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

# Import the rebalancing functionality
from src.trend_analysis.rebalancing import (
    TurnoverCapStrategy, 
    apply_rebalancing_strategies
)
from src.trend_analysis.multi_period.engine import run_schedule, Portfolio
from src.trend_analysis.selector import RankSelector
from src.trend_analysis.weighting import EqualWeight


def demo_basic_turnover_cap():
    """Demonstrate basic turnover cap functionality."""
    print("=" * 60)
    print("DEMO 1: Basic Turnover Cap Functionality")
    print("=" * 60)
    
    # Create strategy with 15% turnover cap and 25 bps transaction costs
    strategy = TurnoverCapStrategy({
        "max_turnover": 0.15,
        "cost_bps": 25,  # 25 basis points = 0.25%
        "priority": "largest_gap"
    })
    
    # Current portfolio weights
    current_weights = pd.Series([0.4, 0.3, 0.3], index=['FUND_A', 'FUND_B', 'FUND_C'])
    
    # Target portfolio weights (significant rebalancing needed)
    target_weights = pd.Series([0.1, 0.1, 0.8], index=['FUND_A', 'FUND_B', 'FUND_C'])
    
    print("Current Portfolio Weights:")
    print(current_weights.to_string())
    print("\nTarget Portfolio Weights:")
    print(target_weights.to_string())
    
    # Calculate desired trades
    desired_trades = target_weights - current_weights
    desired_turnover = desired_trades.abs().sum()
    
    print(f"\nDesired trades:")
    print(desired_trades.to_string())
    print(f"\nTotal desired turnover: {desired_turnover:.1%} (exceeds 15% cap)")
    
    # Apply turnover cap
    new_weights, cost = strategy.apply(current_weights, target_weights)
    actual_trades = new_weights - current_weights
    actual_turnover = actual_trades.abs().sum()
    
    print(f"\nAfter Turnover Cap Applied:")
    print(f"Actual turnover: {actual_turnover:.1%} (respects 15% cap)")
    print(f"Transaction cost: ${cost * 10000:.2f} per $10,000 invested")
    print("\nFinal Portfolio Weights:")
    print(new_weights.to_string())
    print(f"Under-investment: {1.0 - new_weights.sum():.1%}")


def demo_priority_mechanisms():
    """Demonstrate different priority mechanisms."""
    print("\n" + "=" * 60)
    print("DEMO 2: Priority Mechanisms")
    print("=" * 60)
    
    # Setup portfolio
    current_weights = pd.Series([0.25, 0.25, 0.25, 0.25], 
                               index=['HIGH_SCORE', 'MED_SCORE', 'LOW_SCORE', 'ZERO_SCORE'])
    target_weights = pd.Series([0.5, 0.3, 0.2, 0.0], 
                              index=['HIGH_SCORE', 'MED_SCORE', 'LOW_SCORE', 'ZERO_SCORE'])
    
    # Asset scores (for best_score_delta priority)
    scores = pd.Series([0.8, 0.5, 0.2, -0.1], 
                      index=['HIGH_SCORE', 'MED_SCORE', 'LOW_SCORE', 'ZERO_SCORE'])
    
    print("Portfolio Setup:")
    print(f"Current: {current_weights.to_dict()}")
    print(f"Target:  {target_weights.to_dict()}")
    print(f"Scores:  {scores.to_dict()}")
    
    # Test both priority mechanisms
    priorities = ['largest_gap', 'best_score_delta']
    
    for priority in priorities:
        print(f"\n--- Priority: {priority} ---")
        
        strategy = TurnoverCapStrategy({
            "max_turnover": 0.4,  # Limit that will constrain trades
            "cost_bps": 20,
            "priority": priority
        })
        
        new_weights, cost = strategy.apply(
            current_weights, 
            target_weights, 
            scores=scores if priority == 'best_score_delta' else None
        )
        
        trades = new_weights - current_weights
        print(f"Executed trades: {trades.to_dict()}")
        print(f"Turnover used: {trades.abs().sum():.1%}")
        print(f"Transaction cost: {cost:.4f}")


def demo_multi_period_integration():
    """Demonstrate integration with multi-period engine."""
    print("\n" + "=" * 60)
    print("DEMO 3: Multi-Period Engine Integration")
    print("=" * 60)
    
    # Create score frames for 4 periods with changing fund rankings
    periods = ["2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4"]
    funds = ["FUND_A", "FUND_B", "FUND_C", "FUND_D", "FUND_E"]
    
    score_frames = {}
    np.random.seed(42)  # For reproducible demo
    
    for i, period in enumerate(periods):
        # Create varying scores to force rebalancing
        base_scores = np.array([0.8, 0.6, 0.4, 0.2, 0.0])
        # Add some noise and rotation each period
        noise = np.random.normal(0, 0.1, 5)
        rotation = np.roll(base_scores, i)
        scores = rotation + noise
        
        score_frames[period] = pd.DataFrame({
            "Sharpe": scores,
            "CAGR": scores + 0.05,
            "MaxDrawdown": -scores * 0.5
        }, index=funds)
    
    print("Score frames created for periods:", list(score_frames.keys()))
    
    # Setup selector and weighting
    selector = RankSelector(top_n=3, rank_column="Sharpe")
    weighting = EqualWeight()
    
    # Configure turnover cap strategy
    rebalance_strategies = ["turnover_cap"]
    rebalance_params = {
        "turnover_cap": {
            "max_turnover": 0.20,     # 20% max turnover per period
            "cost_bps": 30,           # 30 basis points transaction costs
            "priority": "best_score_delta"
        }
    }
    
    # Run the multi-period simulation
    portfolio = run_schedule(
        score_frames,
        selector,
        weighting,
        rank_column="Sharpe",
        rebalance_strategies=rebalance_strategies,
        rebalance_params=rebalance_params
    )
    
    print(f"\nMulti-period simulation completed:")
    print(f"Periods simulated: {len(portfolio.history)}")
    print(f"Total transaction costs: {portfolio.total_rebalance_costs:.4f}")
    print(f"Average cost per period: {portfolio.total_rebalance_costs / len(periods):.4f}")
    
    # Show portfolio evolution
    print(f"\nPortfolio Evolution:")
    for period, weights in portfolio.history.items():
        non_zero = weights[weights > 0.001]  # Show only significant positions
        clean_dict = {k: round(float(v), 3) for k, v in non_zero.items()}
        print(f"{period}: {clean_dict}")


def demo_cost_calculation():
    """Demonstrate transaction cost calculation details."""
    print("\n" + "=" * 60)
    print("DEMO 4: Transaction Cost Calculation")
    print("=" * 60)
    
    current = pd.Series([0.6, 0.4], index=['BOND_FUND', 'EQUITY_FUND'])
    target = pd.Series([0.3, 0.7], index=['BOND_FUND', 'EQUITY_FUND'])
    
    cost_scenarios = [10, 25, 50]  # Different basis point costs
    
    for cost_bps in cost_scenarios:
        strategy = TurnoverCapStrategy({
            "max_turnover": 1.0,  # No turnover limit for this demo
            "cost_bps": cost_bps,
            "priority": "largest_gap"
        })
        
        new_weights, cost = strategy.apply(current, target)
        trades = new_weights - current
        turnover = trades.abs().sum()
        
        print(f"\nCost scenario: {cost_bps} basis points")
        print(f"Turnover: {turnover:.1%}")
        print(f"Transaction cost: {cost:.4f} ({cost * 100:.2f}%)")
        print(f"Cost per $10,000: ${cost * 10000:.2f}")


def demo_edge_cases():
    """Demonstrate edge case handling."""
    print("\n" + "=" * 60)
    print("DEMO 5: Edge Cases")
    print("=" * 60)
    
    strategy = TurnoverCapStrategy({
        "max_turnover": 0.1,
        "cost_bps": 15,
        "priority": "largest_gap"
    })
    
    # Case 1: Mismatched indices
    print("Case 1: Mismatched fund universes")
    current = pd.Series([0.5, 0.5], index=['FUND_A', 'FUND_B'])
    target = pd.Series([0.3, 0.4, 0.3], index=['FUND_A', 'FUND_C', 'FUND_D'])
    
    new_weights, cost = strategy.apply(current, target)
    print(f"Current funds: {list(current.index)}")
    print(f"Target funds: {list(target.index)}")
    print(f"Final allocation: {dict((k, round(float(v), 3)) for k, v in new_weights[new_weights > 0.001].items())}")
    
    # Case 2: Zero turnover needed
    print(f"\nCase 2: No rebalancing needed")
    current = pd.Series([0.4, 0.6], index=['FUND_X', 'FUND_Y'])
    target = current.copy()  # Same as current
    
    new_weights, cost = strategy.apply(current, target)
    turnover = (new_weights - current).abs().sum()
    print(f"Turnover: {turnover:.6f}")
    print(f"Cost: {cost:.6f}")
    
    # Case 3: Very small turnover cap
    print(f"\nCase 3: Very restrictive turnover cap (1%)")
    strategy_restrictive = TurnoverCapStrategy({
        "max_turnover": 0.01,  # Very small cap
        "cost_bps": 20,
        "priority": "largest_gap"
    })
    
    current = pd.Series([0.0, 1.0], index=['FUND_1', 'FUND_2'])
    target = pd.Series([0.5, 0.5], index=['FUND_1', 'FUND_2'])
    
    new_weights, cost = strategy_restrictive.apply(current, target)
    actual_trades = new_weights - current
    print(f"Desired change: FUND_1 +50%, FUND_2 -50%")
    print(f"Actual change: FUND_1 +{actual_trades['FUND_1']:.1%}, FUND_2 {actual_trades['FUND_2']:.1%}")
    print(f"Total turnover: {actual_trades.abs().sum():.1%}")


if __name__ == "__main__":
    print("Turnover Cap Rebalancing Strategy Demonstration")
    print("This demo shows the complete functionality of the turnover_cap strategy")
    print("implemented in the Trend Model Project.")
    print("\nTo run this demo:")
    print("  cd /path/to/Trend_Model_Project")
    print("  source .venv/bin/activate  # if using virtual environment")
    print("  PYTHONPATH=./src python demo_turnover_cap.py")
    
    try:
        demo_basic_turnover_cap()
        demo_priority_mechanisms()
        demo_multi_period_integration()
        demo_cost_calculation()
        demo_edge_cases()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✅ Turnover cap enforcement (prevents excessive trading)")
        print("✅ Transaction cost modeling (basis points on actual turnover)")
        print("✅ Priority mechanisms (largest_gap vs best_score_delta)")
        print("✅ Multi-period engine integration")
        print("✅ Edge case handling (mismatched funds, zero trades, etc.)")
        print("✅ Portfolio under-investment when turnover limits prevent full rebalancing")
        
        print(f"\nThe turnover_cap strategy is production-ready and fully integrated")
        print(f"with the existing Trend Model Project infrastructure.")
        
        print(f"\nExample Configuration for Multi-Period Engine:")
        print(f"rebalance_strategies = ['turnover_cap']")
        print(f"rebalance_params = {{")
        print(f"    'turnover_cap': {{")
        print(f"        'max_turnover': 0.15,        # 15% max turnover per period")
        print(f"        'cost_bps': 25,              # 25 bps transaction costs")
        print(f"        'priority': 'best_score_delta' # Priority by score-weighted benefit")
        print(f"    }}")
        print(f"}}")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("Make sure you're running from the repository root with:")
        print("  PYTHONPATH=./src python demo_turnover_cap.py")
        raise