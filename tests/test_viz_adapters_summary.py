from __future__ import annotations

import pandas as pd

from trend_analysis.viz.adapters import SUMMARY_REQUIRED_COLUMNS, make_summary


def _sample_results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"fold_id": 1, "fold_label": "Fold 1", "path_id": 1, "strategy": "A", "sharpe": 1.0},
            {"fold_id": 1, "fold_label": "Fold 1", "path_id": 2, "strategy": "A", "sharpe": 1.4},
            {"fold_id": 1, "fold_label": "Fold 1", "path_id": 3, "strategy": "B", "sharpe": 0.8},
            {"fold_id": 2, "fold_label": "Fold 2", "path_id": 1, "strategy": "A", "sharpe": 0.9},
        ]
    )


def test_make_summary_required_columns_and_expected_shape() -> None:
    summary = make_summary(_sample_results_frame())

    for column in SUMMARY_REQUIRED_COLUMNS:
        assert column in summary.columns
    assert "sharpe" in summary.columns
    assert summary.shape == (3, 5)

    rows = summary.sort_values(["fold_id", "strategy"]).reset_index(drop=True)
    assert rows["paths"].tolist() == [2, 1, 1]
    assert rows["sharpe"].tolist() == [1.2, 0.8, 0.9]
