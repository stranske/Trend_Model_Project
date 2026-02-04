from __future__ import annotations

import pandas as pd

from trend_analysis.monte_carlo.results import (
    RESULT_BASE_COLUMNS,
    MonteCarloResults,
    StrategyEvaluation,
    build_cross_fold_summary_frame,
    build_pooled_summary_frame,
    build_results_frame,
    build_summary_frame,
    export_results,
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


def test_build_results_frame_missing_fold_id_keeps_column() -> None:
    evaluation = StrategyEvaluation(
        fold_id=None,
        path_id=9,
        strategy_name="StrategyB",
        metrics={"alpha": 2.5},
        metric_source="unit_test",
        path_hash="hash-2",
        seed=456,
    )

    frame = build_results_frame([evaluation])

    assert "fold_id" in frame.columns
    assert pd.isna(frame.loc[0, "fold_id"])


def test_build_summary_frame_groups_by_fold_id() -> None:
    frame = pd.DataFrame(
        [
            {"fold_id": 1, "path_id": 1, "strategy": "A", "metric": 1.0},
            {"fold_id": 1, "path_id": 2, "strategy": "A", "metric": 3.0},
            {"fold_id": 2, "path_id": 3, "strategy": "A", "metric": 2.0},
            {"fold_id": 2, "path_id": 4, "strategy": "B", "metric": 4.0},
        ]
    )

    summary = build_summary_frame(frame).sort_values(["fold_id", "strategy"]).reset_index(drop=True)

    assert list(summary.columns[:2]) == ["fold_id", "strategy"]
    assert summary.loc[0, "paths"] == 2
    assert summary.loc[0, "metric"] == 2.0


def test_build_summary_frame_empty_includes_fold_id_if_present() -> None:
    frame = pd.DataFrame(columns=list(RESULT_BASE_COLUMNS))

    summary = build_summary_frame(frame)

    assert list(summary.columns) == ["fold_id", "strategy", "paths"]
    assert summary.empty


def test_build_pooled_summary_frame_ignores_folds() -> None:
    frame = pd.DataFrame(
        [
            {"fold_id": 1, "path_id": 1, "strategy": "A", "metric": 1.0},
            {"fold_id": 1, "path_id": 2, "strategy": "A", "metric": 3.0},
            {"fold_id": 2, "path_id": 3, "strategy": "A", "metric": 5.0},
            {"fold_id": 2, "path_id": 4, "strategy": "A", "metric": 7.0},
        ]
    )

    pooled = build_pooled_summary_frame(frame)

    assert pooled.loc[0, "scope"] == "pooled"
    assert pooled.loc[0, "pooled_scope"] == "summary"
    assert "fold_id" in pooled.columns
    assert pd.isna(pooled.loc[0, "fold_id"])
    assert pooled.loc[0, "strategy"] == "A"
    assert pooled.loc[0, "metric"] == 4.0
    assert pooled.loc[0, "paths"] == 4
    assert pooled.loc[0, "folds"] == 2


def test_build_pooled_summary_frame_empty_returns_columns() -> None:
    pooled = build_pooled_summary_frame(pd.DataFrame())

    assert list(pooled.columns) == [
        "scope",
        "pooled_scope",
        "fold_id",
        "strategy",
        "paths",
        "folds",
    ]
    assert pooled.empty


def test_build_cross_fold_summary_frame_reports_fold_stats() -> None:
    frame = pd.DataFrame(
        [
            {"fold_id": 1, "path_id": 1, "strategy": "A", "metric": 1.0, "metric2": 2.0},
            {"fold_id": 1, "path_id": 2, "strategy": "A", "metric": 3.0, "metric2": 6.0},
            {"fold_id": 2, "path_id": 3, "strategy": "A", "metric": 5.0, "metric2": 10.0},
            {"fold_id": 2, "path_id": 4, "strategy": "A", "metric": 7.0, "metric2": 14.0},
            {"fold_id": 3, "path_id": 5, "strategy": "A", "metric": 9.0, "metric2": 18.0},
            {"fold_id": 3, "path_id": 6, "strategy": "A", "metric": 11.0, "metric2": 22.0},
        ]
    )

    cross_fold = build_cross_fold_summary_frame(frame)

    assert cross_fold.loc[0, "scope"] == "cross_fold"
    assert "fold_id" in cross_fold.columns
    assert pd.isna(cross_fold.loc[0, "fold_id"])
    assert cross_fold.loc[0, "strategy"] == "A"
    assert cross_fold.loc[0, "folds"] == 3
    assert cross_fold.loc[0, "metric_mean"] == 6.0
    assert cross_fold.loc[0, "metric_std"] == 4.0
    assert cross_fold.loc[0, "metric_min"] == 2.0
    assert cross_fold.loc[0, "metric_max"] == 10.0
    assert cross_fold.loc[0, "metric_median"] == 6.0
    assert cross_fold.loc[0, "metric2_mean"] == 12.0
    assert cross_fold.loc[0, "metric2_std"] == 8.0
    assert cross_fold.loc[0, "metric2_min"] == 4.0
    assert cross_fold.loc[0, "metric2_max"] == 20.0
    assert cross_fold.loc[0, "metric2_median"] == 12.0
    assert cross_fold.loc[0, "paths_mean"] == 2.0
    assert cross_fold.loc[0, "paths_std"] == 0.0
    assert cross_fold.loc[0, "paths_min"] == 2.0
    assert cross_fold.loc[0, "paths_max"] == 2.0
    assert cross_fold.loc[0, "paths_median"] == 2.0


def test_build_cross_fold_summary_frame_empty_without_fold_id() -> None:
    frame = pd.DataFrame(
        [
            {"path_id": 1, "strategy": "A", "metric": 1.0},
            {"path_id": 2, "strategy": "A", "metric": 2.0},
        ]
    )

    cross_fold = build_cross_fold_summary_frame(frame)

    assert list(cross_fold.columns) == ["scope", "fold_id", "strategy", "folds"]
    assert cross_fold.empty


def test_export_results_writes_pooled_summary(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {"fold_id": 1, "path_id": 1, "strategy": "A", "metric": 1.0},
            {"fold_id": 1, "path_id": 2, "strategy": "A", "metric": 3.0},
            {"fold_id": 2, "path_id": 3, "strategy": "A", "metric": 5.0},
        ]
    )
    summary = build_summary_frame(frame)
    pooled = build_pooled_summary_frame(frame)
    cross_fold = build_cross_fold_summary_frame(frame)
    results = MonteCarloResults(
        mode="two_layer",
        evaluations=[],
        errors=[],
        results_frame=frame,
        summary_frame=summary,
        cross_fold_summary_frame=cross_fold,
        pooled_summary_frame=pooled,
        metadata={},
    )

    exported = export_results(results, tmp_path, formats=["csv"])

    results_path = exported["results_csv"]
    results_frame = pd.read_csv(results_path)
    assert "fold_id" in results_frame.columns
    assert set(results_frame["fold_id"].tolist()) == {1, 2}

    summary_path = exported["summary_csv"]
    summary_frame = pd.read_csv(summary_path)
    assert "fold_id" in summary_frame.columns
    assert set(summary_frame["fold_id"].tolist()) == {1, 2}

    cross_path = exported["cross_fold_summary_csv"]
    cross_frame = pd.read_csv(cross_path)
    assert cross_frame.loc[0, "scope"] == "cross_fold"
    assert "fold_id" in cross_frame.columns
    assert pd.isna(cross_frame.loc[0, "fold_id"])

    pooled_path = exported["pooled_summary_csv"]
    assert pooled_path.exists()
    pooled_frame = pd.read_csv(pooled_path)
    assert pooled_frame.loc[0, "scope"] == "pooled"
    assert pooled_frame.loc[0, "pooled_scope"] == "summary"
    assert "fold_id" in pooled_frame.columns
    assert pd.isna(pooled_frame.loc[0, "fold_id"])
    assert pooled_frame.loc[0, "paths"] == 3
    assert pooled_frame.loc[0, "folds"] == 2


def test_export_results_skips_pooled_summary_when_missing(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {"fold_id": 1, "path_id": 1, "strategy": "A", "metric": 1.0},
            {"fold_id": 1, "path_id": 2, "strategy": "A", "metric": 3.0},
        ]
    )
    summary = build_summary_frame(frame)
    cross_fold = build_cross_fold_summary_frame(frame)
    results = MonteCarloResults(
        mode="two_layer",
        evaluations=[],
        errors=[],
        results_frame=frame,
        summary_frame=summary,
        cross_fold_summary_frame=cross_fold,
        pooled_summary_frame=None,
        metadata={},
    )

    exported = export_results(results, tmp_path, formats=["csv"])

    assert "pooled_summary_csv" not in exported
