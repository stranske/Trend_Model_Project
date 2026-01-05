"""Test regex optimization in rank_selection module."""

import time

import pandas as pd

from trend_analysis.core import rank_selection as rs


def test_regex_performance_through_rank_select():
    """Test regex performance by calling rank_select_funds multiple times."""
    # Create a DataFrame with many fund names to trigger regex calls
    fund_names = [
        "Goldman Sachs Fund A",
        "JP_Morgan-Global",
        "VANGUARD small cap",
        "Fidelity123 Growth",
        "BlackRock Strategic",
        "STATE STREET Corp",
        "T ROWE PRICE Equity",
        "CAPITAL GROUP Value",
        "Wellington Fund",
        "American Century Growth",
        "Janus Henderson",
        "Franklin Templeton",
    ]

    # Create a test DataFrame
    df = pd.DataFrame({name: [0.01 + i * 0.001 for i in range(6)] for name in fund_names})

    cfg = rs.RiskStatsConfig(risk_free=0.0)

    # Measure performance by calling rank_select_funds multiple times
    iterations = 50
    start_time = time.time()

    for _ in range(iterations):
        selected = rs.rank_select_funds(
            df,
            cfg,
            inclusion_approach="top_n",
            n=5,
            score_by="AnnualReturn",
            limit_one_per_firm=True,
        )
        # Just verify we get results
        assert len(selected) > 0

    end_time = time.time()
    execution_time = end_time - start_time

    print(
        "Optimized performance: "
        f"{execution_time:.4f}s for {iterations} iterations with "
        f"{len(fund_names)} funds"
    )

    # Performance should be reasonable - the optimization should provide fast execution
    assert execution_time < 5.0, f"Performance too slow: {execution_time:.4f}s"


def test_deduplication_behavior():
    """Test fund deduplication behavior to understand current logic."""
    # Create a DataFrame with funds that should be considered from same firms
    df = pd.DataFrame(
        {
            "Goldman Sachs Fund A": [0.05, 0.06, 0.04, 0.05, 0.04, 0.03],
            "Goldman-Sachs Fund B": [0.04, 0.05, 0.03, 0.04, 0.03, 0.02],
            "Vanguard Fund X": [0.03, 0.04, 0.02, 0.03, 0.02, 0.01],
            "VANGUARD Fund Y": [0.02, 0.03, 0.01, 0.02, 0.01, 0.00],
        }
    )

    cfg = rs.RiskStatsConfig(risk_free=0.0)

    # Test with limit_one_per_firm=True (default)
    selected = rs.rank_select_funds(
        df,
        cfg,
        inclusion_approach="top_n",
        n=4,  # Request all funds to see what happens
        score_by="AnnualReturn",
        limit_one_per_firm=True,
    )

    print(f"Selected funds with deduplication: {selected}")

    # Test with limit_one_per_firm=False
    selected_all = rs.rank_select_funds(
        df,
        cfg,
        inclusion_approach="top_n",
        n=4,
        score_by="AnnualReturn",
        limit_one_per_firm=False,
    )

    print(f"Selected funds without deduplication: {selected_all}")

    # Basic validation - should get fewer or equal funds with deduplication
    assert len(selected) <= len(selected_all)
