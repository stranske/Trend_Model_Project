from __future__ import annotations

import pandas as pd

from trend_analysis.monte_carlo.results import (
    RESULT_BASE_COLUMNS,
    StrategyEvaluation,
    build_results_frame,
    build_summary_frame,
)


def test_build_results_frame_includes_fold_id_and_base_columns() -> None:
    evaluation = StrategyEvaluation(
        fold_id=2,
        path_id=7,
        strategy_name="StrategyA",
        metrics={"alpha": 1.5, "beta": 0.8},
        metric_source="unit_test",
        path_hash="hash",
        seed=123,
    )

    frame = build_results_frame([evaluation])

    assert "fold_id" in frame.columns
    assert list(frame.columns[: len(RESULT_BASE_COLUMNS)]) == list(RESULT_BASE_COLUMNS)
    assert frame.loc[0, "fold_id"] == 2
    assert frame.loc[0, "path_id"] == 7
    assert frame.loc[0, "strategy"] == "StrategyA"


def test_build_results_frame_empty_has_base_columns() -> None:
    frame = build_results_frame([])

    assert list(frame.columns) == list(RESULT_BASE_COLUMNS)
    assert frame.empty


def test_build_summary_frame_groups_by_fold_id() -> None:
    frame = pd.DataFrame(
        [
            {"fold_id": 1, "path_id": 1, "strategy": "A", "metric": 1.0},
            {"fold_id": 1, "path_id": 2, "strategy": "A", "metric": 3.0},
            {"fold_id": 2, "path_id": 3, "strategy": "A", "metric": 2.0},
            {"fold_id": 2, "path_id": 4, "strategy": "B", "metric": 4.0},
        ]
    )

    summary = build_summary_frame(frame).sort_values(["fold_id", "strategy"]).reset_index(
        drop=True
    )

    assert list(summary.columns[:2]) == ["fold_id", "strategy"]
    assert summary.loc[0, "paths"] == 2
    assert summary.loc[0, "metric"] == 2.0


def test_build_summary_frame_empty_includes_fold_id_if_present() -> None:
    frame = pd.DataFrame(columns=list(RESULT_BASE_COLUMNS))

    summary = build_summary_frame(frame)

    assert list(summary.columns) == ["fold_id", "strategy", "paths"]
    assert summary.empty
