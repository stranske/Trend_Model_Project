import warnings

import pandas as pd
import pytest

from trend_analysis.core.rank_selection import (
    RankSelectionDiagnostics,
    RiskStatsConfig,
    rank_select_funds,
)


def test_rank_select_funds_warns_and_reports_on_empty_scores():
    df = pd.DataFrame()
    cfg = RiskStatsConfig()

    with pytest.warns(RuntimeWarning) as caught:
        selected, diagnostics = rank_select_funds(
            df,
            cfg,
            inclusion_approach="top_n",
            n=1,
            score_by="AnnualReturn",
            return_diagnostics=True,
        )

    assert selected == []
    assert isinstance(diagnostics, RankSelectionDiagnostics)
    assert diagnostics.reason == "No candidate columns available for ranking"
    assert diagnostics.non_null_scores == 0
    assert diagnostics.total_candidates == 0
    assert any(
        diagnostics.reason in str(warning.message) for warning in caught.list
    )


def test_rank_select_funds_reports_threshold_filtering():
    df = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]})
    cfg = RiskStatsConfig()

    with pytest.warns(RuntimeWarning):
        selected, diagnostics = rank_select_funds(
            df,
            cfg,
            inclusion_approach="threshold",
            threshold=0.5,
            score_by="AnnualReturn",
            return_diagnostics=True,
        )

    assert selected == []
    assert isinstance(diagnostics, RankSelectionDiagnostics)
    assert diagnostics.reason == "All candidate scores filtered out by threshold"
    assert diagnostics.threshold == 0.5
    assert diagnostics.non_null_scores == 0


def test_rank_select_funds_success_has_no_diagnostics():
    df = pd.DataFrame(
        {
            "A": [0.02, 0.03, -0.01, 0.04],
            "B": [0.01, 0.02, 0.0, 0.03],
        }
    )
    cfg = RiskStatsConfig(risk_free=0.0)

    with warnings.catch_warnings(record=True) as caught:
        selected, diagnostics = rank_select_funds(
            df,
            cfg,
            inclusion_approach="top_n",
            n=1,
            score_by="AnnualReturn",
            return_diagnostics=True,
        )

    assert selected == ["A"]
    assert diagnostics is None
    assert caught == []
