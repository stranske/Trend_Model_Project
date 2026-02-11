from __future__ import annotations

import pandas as pd

from trend_analysis.viz.adapters import (
    SUMMARY_REQUIRED_COLUMNS,
    SUMMARY_REQUIRED_DTYPES,
    make_summary,
)


def _sample_results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold_id": 1,
                "fold_label": "Fold 1",
                "path_id": 1,
                "strategy": "A",
                "sharpe": 1.0,
                "max_drawdown": -0.20,
            },
            {
                "fold_id": 1,
                "fold_label": "Fold 1",
                "path_id": 2,
                "strategy": "A",
                "sharpe": 1.4,
                "max_drawdown": -0.30,
            },
            {
                "fold_id": 2,
                "fold_label": "Fold 2",
                "path_id": 1,
                "strategy": "A",
                "sharpe": 0.8,
                "max_drawdown": -0.10,
            },
            {
                "fold_id": 2,
                "fold_label": "Fold 2",
                "path_id": 2,
                "strategy": "B",
                "sharpe": 0.5,
                "max_drawdown": -0.40,
            },
        ]
    )


def test_make_summary_includes_required_schema_and_dtypes() -> None:
    summary = make_summary(_sample_results_frame())

    for col in SUMMARY_REQUIRED_COLUMNS:
        assert col in summary.columns
    assert summary.dtypes["fold_id"] == SUMMARY_REQUIRED_DTYPES["fold_id"]
    assert summary.dtypes["fold_label"] == SUMMARY_REQUIRED_DTYPES["fold_label"]
    assert summary.dtypes["strategy"] == SUMMARY_REQUIRED_DTYPES["strategy"]
    assert summary.dtypes["paths"] == SUMMARY_REQUIRED_DTYPES["paths"]
    assert "sharpe" in summary.columns
    assert "max_drawdown" in summary.columns
    assert summary.dtypes["sharpe"] == "float64"
    assert summary.dtypes["max_drawdown"] == "float64"


def test_make_summary_fold_selection_filters_by_fold_id() -> None:
    summary = make_summary(_sample_results_frame(), fold_selection=1)

    assert summary["fold_id"].dropna().unique().tolist() == [1]
    assert summary["paths"].tolist() == [2]
    assert summary["sharpe"].tolist() == [1.2]


def test_make_summary_fold_selection_filters_by_fold_label() -> None:
    summary = make_summary(_sample_results_frame(), fold_selection="Fold 2")

    assert set(summary["strategy"].tolist()) == {"A", "B"}
    assert summary["fold_label"].dropna().unique().tolist() == ["Fold 2"]
    rows = summary.sort_values("strategy").reset_index(drop=True)
    assert rows.loc[0, "sharpe"] == 0.8
    assert rows.loc[1, "sharpe"] == 0.5


def test_make_summary_pooled_aggregates_across_folds() -> None:
    summary = make_summary(_sample_results_frame(), fold_selection="pooled")

    assert set(summary["strategy"].tolist()) == {"A", "B"}
    row_a = summary[summary["strategy"] == "A"].iloc[0]
    row_b = summary[summary["strategy"] == "B"].iloc[0]
    assert pd.isna(row_a["fold_id"])
    assert pd.isna(row_a["fold_label"])
    assert row_a["paths"] == 3
    assert row_a["sharpe"] == 1.0666666666666667
    assert row_b["paths"] == 1
    assert row_b["sharpe"] == 0.5
