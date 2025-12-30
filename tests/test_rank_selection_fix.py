import pandas as pd

from trend_analysis.core.rank_selection import RiskStatsConfig, rank_select_funds


def test_rank_selection_sorts_correctly():
    """Test that rank_select_funds selects actual top performers by metric."""

    # Create test data where best performers are NOT first columns
    data = {
        "Mgr_01": [0.01, 0.02, 0.01],  # Poor performance
        "Mgr_02": [0.02, 0.03, 0.02],  # Medium performance
        "Mgr_03": [0.05, 0.06, 0.05],  # HIGH performance - should be selected
        "Mgr_04": [0.04, 0.05, 0.04],  # HIGH performance - should be selected
        "Mgr_05": [0.01, 0.01, 0.01],  # Poor performance
    }
    df = pd.DataFrame(data)
    cfg = RiskStatsConfig()

    # Select top 2 by Sharpe - should get Mgr_03 and Mgr_04
    selected = rank_select_funds(df, cfg, inclusion_approach="top_n", n=2, score_by="Sharpe")

    # Verify we get actual top performers, not just first 2 columns
    assert len(selected) == 2
    assert "Mgr_03" in selected  # Best performer
    assert "Mgr_04" in selected  # Second best performer
    assert "Mgr_01" not in selected
    assert "Mgr_02" not in selected


def test_rank_selection_ascending_metric():
    """Test ascending metrics (MaxDrawdown) handled correctly."""

    data = {
        "Mgr_01": [-0.05, -0.04, -0.03],  # Bad drawdown
        "Mgr_02": [-0.01, -0.02, -0.01],  # Good drawdown
        "Mgr_03": [-0.03, -0.04, -0.02],  # Medium drawdown
        "Mgr_04": [-0.01, -0.01, -0.005],  # Best drawdown
    }
    df = pd.DataFrame(data)
    cfg = RiskStatsConfig()

    selected = rank_select_funds(df, cfg, inclusion_approach="top_n", n=2, score_by="MaxDrawdown")

    # Should select funds with lowest (best) drawdowns
    assert len(selected) == 2
    assert "Mgr_04" in selected  # Best drawdown
    assert "Mgr_02" in selected  # Second best drawdown
    assert "Mgr_01" not in selected  # Worst drawdown
