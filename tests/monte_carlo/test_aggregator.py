from __future__ import annotations

import pandas as pd
import pytest

import trend_analysis.monte_carlo.export as export_module
from trend_analysis.monte_carlo.aggregator import (
    AGGREGATION_PATH_COLUMNS,
    BREACH_COLUMNS,
    EXPECTED_SHORTFALL_COLUMNS,
    PATH_COLUMNS,
    QUANTILE_COLUMNS,
    MonteCarloAggregationResults,
    aggregate_monte_carlo_results,
    aggregation_frame_schemas,
    breach_frame_schema,
    build_breach_frame,
    build_expected_shortfall_frame,
    build_path_frame,
    build_quantiles_frame,
    expected_shortfall_frame_schema,
    path_frame_schema,
    quantiles_frame_schema,
)
from trend_analysis.monte_carlo.export import export_aggregation_results


def _sample_results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold_id": 1,
                "path_id": 1,
                "strategy": "A",
                "metric": 1.0,
                "metric2": 2.0,
            },
            {
                "fold_id": 1,
                "path_id": 2,
                "strategy": "A",
                "metric": 3.0,
                "metric2": 4.0,
            },
            {
                "fold_id": 1,
                "path_id": 3,
                "strategy": "A",
                "metric": 5.0,
                "metric2": 6.0,
            },
        ]
    )


def test_build_path_frame_schema_and_values() -> None:
    results_frame = _sample_results_frame()

    path_frame = build_path_frame(results_frame)

    assert list(path_frame.columns[: len(AGGREGATION_PATH_COLUMNS)]) == list(
        AGGREGATION_PATH_COLUMNS
    )
    assert path_frame.loc[0, "strategy"] == "A"
    assert path_frame.loc[0, "path"] == 1
    assert path_frame.loc[0, "fold"] == 1
    assert path_frame.loc[0, "metric"] == pytest.approx(1.0)


def test_build_path_frame_excludes_path_and_fold_from_metrics() -> None:
    results_frame = pd.DataFrame(
        [
            {"fold": 0, "path": 10, "strategy": "A", "metric": 1.0},
            {"fold": 0, "path": 11, "strategy": "A", "metric": 2.0},
        ]
    )

    path_frame = build_path_frame(results_frame)

    assert "metric" in path_frame.columns
    assert "path" in path_frame.columns
    assert "fold" in path_frame.columns
    assert list(path_frame.columns).count("path") == 1
    assert list(path_frame.columns).count("fold") == 1


def test_build_path_frame_excludes_numeric_strategy_from_metrics() -> None:
    results_frame = pd.DataFrame(
        [
            {"fold_id": 0, "path_id": 10, "strategy": 1, "metric": 1.0},
            {"fold_id": 0, "path_id": 11, "strategy": 1, "metric": 2.0},
        ]
    )

    path_frame = build_path_frame(results_frame)
    quantiles = build_quantiles_frame(path_frame, [0.5])

    assert list(path_frame.columns).count("strategy") == 1
    assert "metric" in path_frame.columns
    assert set(quantiles["metric"]) == {"metric"}


def test_build_path_frame_excludes_seed_and_ids_from_metrics() -> None:
    results_frame = pd.DataFrame(
        [
            {
                "strategy": "A",
                "fold_id": 1,
                "path_id": 10,
                "seed": 123,
                "metric": 1.0,
            }
        ]
    )

    path_frame = build_path_frame(results_frame)

    assert list(path_frame.columns) == list(AGGREGATION_PATH_COLUMNS) + ["metric"]
    assert "seed" not in path_frame.columns
    assert "fold_id" not in path_frame.columns
    assert "path_id" not in path_frame.columns


def test_build_path_frame_excludes_paths_and_folds_from_metrics() -> None:
    results_frame = pd.DataFrame(
        [
            {
                "strategy": "A",
                "fold_id": 1,
                "path_id": 10,
                "paths": 20,
                "folds": 3,
                "metric": 1.0,
            }
        ]
    )

    path_frame = build_path_frame(results_frame)

    assert list(path_frame.columns) == list(AGGREGATION_PATH_COLUMNS) + ["metric"]


def test_build_path_frame_excludes_metadata_columns_from_metrics() -> None:
    results_frame = pd.DataFrame(
        [
            {
                "strategy": "A",
                "fold_id": 1,
                "path_id": 10,
                "fold_label": "1",
                "path_hash": "2",
                "metric_source": "3",
                "metric": 1.0,
            }
        ]
    )

    path_frame = build_path_frame(results_frame)

    assert list(path_frame.columns) == list(AGGREGATION_PATH_COLUMNS) + ["metric"]


def test_build_path_frame_preserves_metric_schema_on_empty_input() -> None:
    results_frame = pd.DataFrame(
        {
            "strategy": pd.Series(dtype=str),
            "path": pd.Series(dtype=int),
            "fold": pd.Series(dtype=int),
            "metric": pd.Series(dtype=float),
        }
    )

    path_frame = build_path_frame(results_frame)

    assert list(path_frame.columns) == list(AGGREGATION_PATH_COLUMNS) + ["metric"]


def test_build_path_frame_includes_numeric_string_metrics() -> None:
    results_frame = pd.DataFrame(
        [
            {"strategy": "A", "path": 1, "fold": 0, "metric": "1.5", "note": "x"},
            {"strategy": "A", "path": 2, "fold": 0, "metric": "2.5", "note": "y"},
        ]
    )

    path_frame = build_path_frame(results_frame)

    assert "metric" in path_frame.columns
    assert "note" not in path_frame.columns


def test_build_path_frame_coerces_string_metrics_to_numeric() -> None:
    results_frame = pd.DataFrame(
        [
            {"strategy": "A", "path": 1, "fold": 0, "metric": "1.5"},
            {"strategy": "A", "path": 2, "fold": 0, "metric": "2.5"},
        ]
    )

    path_frame = build_path_frame(results_frame)

    assert pd.api.types.is_numeric_dtype(path_frame["metric"])
    assert path_frame.loc[0, "metric"] == pytest.approx(1.5)


def test_build_breach_frame_coerces_numeric_columns() -> None:
    path_frame = pd.DataFrame(
        [
            {"strategy": "A", "fold": 0, "path": 1, "metric": "1.0"},
            {"strategy": "A", "fold": 0, "path": 2, "metric": "2.0"},
        ]
    )

    breach_frame = build_breach_frame(path_frame, ["1.5"])

    assert pd.api.types.is_numeric_dtype(breach_frame["threshold"])
    assert pd.api.types.is_float_dtype(breach_frame["breach_probability"])
    assert pd.api.types.is_integer_dtype(breach_frame["paths"])


def test_build_breach_frame_rejects_boolean_thresholds() -> None:
    path_frame = pd.DataFrame(
        [
            {"strategy": "A", "fold": 0, "path": 1, "metric": 1.0},
            {"strategy": "A", "fold": 0, "path": 2, "metric": 2.0},
        ]
    )

    with pytest.raises(TypeError, match="Breach thresholds must be numeric values"):
        build_breach_frame(path_frame, True)


def test_build_expected_shortfall_frame_coerces_numeric_columns() -> None:
    path_frame = pd.DataFrame(
        [
            {"strategy": "A", "fold": 0, "path": 1, "metric": "1.0"},
            {"strategy": "A", "fold": 0, "path": 2, "metric": "3.0"},
        ]
    )

    shortfall_frame = build_expected_shortfall_frame(
        path_frame, {"metric": {"alpha": "0.5", "tail": "upper"}}
    )

    assert pd.api.types.is_numeric_dtype(shortfall_frame["alpha"])
    assert pd.api.types.is_numeric_dtype(shortfall_frame["threshold"])
    assert pd.api.types.is_numeric_dtype(shortfall_frame["expected_shortfall"])
    assert pd.api.types.is_integer_dtype(shortfall_frame["paths"])


def test_build_path_frame_fills_missing_strategy() -> None:
    results_frame = pd.DataFrame(
        [
            {"fold": 0, "path": 10, "metric": 1.0},
            {"fold": 0, "path": 11, "metric": 2.0},
        ]
    )

    path_frame = build_path_frame(results_frame)

    assert list(path_frame.columns[: len(AGGREGATION_PATH_COLUMNS)]) == list(
        AGGREGATION_PATH_COLUMNS
    )
    assert path_frame["strategy"].isna().all()


def test_build_path_frame_uses_strategy_name_column() -> None:
    results_frame = pd.DataFrame(
        [
            {"fold_id": 0, "path_id": 10, "strategy_name": "A", "metric": 1.0},
            {"fold_id": 0, "path_id": 11, "strategy_name": "A", "metric": 2.0},
        ]
    )

    path_frame = build_path_frame(results_frame)

    assert list(path_frame.columns[: len(AGGREGATION_PATH_COLUMNS)]) == list(
        AGGREGATION_PATH_COLUMNS
    )
    assert "strategy_name" not in path_frame.columns
    assert path_frame.loc[0, "strategy"] == "A"
    assert path_frame.loc[0, "metric"] == pytest.approx(1.0)


def test_build_path_frame_prefers_strategy_name_for_numeric_strategy() -> None:
    results_frame = pd.DataFrame(
        [
            {
                "fold_id": 0,
                "path_id": 10,
                "strategy": 1,
                "strategy_name": "Alpha",
                "metric": 1.0,
            },
            {
                "fold_id": 0,
                "path_id": 11,
                "strategy": 1,
                "strategy_name": "Alpha",
                "metric": 2.0,
            },
        ]
    )

    path_frame = build_path_frame(results_frame)

    assert list(path_frame.columns[: len(AGGREGATION_PATH_COLUMNS)]) == list(
        AGGREGATION_PATH_COLUMNS
    )
    assert path_frame.loc[0, "strategy"] == "Alpha"
    assert path_frame.loc[0, "metric"] == pytest.approx(1.0)


def test_build_path_frame_fills_missing_path_and_fold() -> None:
    results_frame = pd.DataFrame(
        [
            {"strategy": "A", "metric": 1.0},
            {"strategy": "A", "metric": 2.0},
        ]
    )

    path_frame = build_path_frame(results_frame)

    assert list(path_frame.columns[: len(AGGREGATION_PATH_COLUMNS)]) == list(
        AGGREGATION_PATH_COLUMNS
    )
    assert path_frame["path"].isna().all()
    assert path_frame["fold"].isna().all()


def test_build_path_frame_uses_fold_label_when_fold_id_missing() -> None:
    results_frame = pd.DataFrame(
        [
            {"strategy": "A", "path_id": 10, "fold_label": "Fold-1", "metric": 1.0},
            {"strategy": "A", "path_id": 11, "fold_label": "Fold-1", "metric": 2.0},
        ]
    )

    path_frame = build_path_frame(results_frame)

    assert path_frame.loc[0, "fold"] == "Fold-1"


def test_build_path_frame_uses_path_hash_when_path_id_missing() -> None:
    results_frame = pd.DataFrame(
        [
            {"strategy": "A", "fold_id": 1, "path_hash": "hash-1", "metric": 1.0},
            {"strategy": "A", "fold_id": 1, "path_hash": "hash-2", "metric": 2.0},
        ]
    )

    path_frame = build_path_frame(results_frame)

    assert path_frame.loc[0, "path"] == "hash-1"


def test_path_frame_schema_includes_numeric_metrics_only() -> None:
    results_frame = pd.DataFrame(
        [
            {
                "strategy": "A",
                "path": 1,
                "fold": 0,
                "metric": 1.0,
                "metric_str": "2.0",
                "note": "x",
            }
        ]
    )

    schema = path_frame_schema(results_frame)

    assert schema == tuple(AGGREGATION_PATH_COLUMNS) + ("metric", "metric_str")


def test_build_path_frame_sorts_mixed_strategy_types() -> None:
    results_frame = pd.DataFrame(
        [
            {"strategy": "A", "path": 2, "fold": 0, "metric": 1.0},
            {"strategy": 1, "path": 1, "fold": 0, "metric": 2.0},
        ]
    )

    path_frame = build_path_frame(results_frame)

    assert list(path_frame["strategy"]) == [1, "A"]
    assert list(path_frame["path"]) == [1, 2]


def test_build_quantiles_frame_reports_requested_quantiles() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    quantiles = build_quantiles_frame(path_frame, [0.5])

    assert list(quantiles.columns) == list(quantiles_frame_schema())
    assert quantiles.loc[0, "quantile"] == pytest.approx(0.5)
    assert quantiles.loc[0, "metric"] == "metric"
    assert quantiles.loc[0, "value"] == pytest.approx(3.0)
    assert quantiles.loc[0, "paths"] == 3


def test_build_quantiles_frame_coerces_quantile_and_paths_types() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    quantiles = build_quantiles_frame(path_frame, [0.5])

    assert pd.api.types.is_numeric_dtype(quantiles["quantile"])
    assert pd.api.types.is_numeric_dtype(quantiles["value"])
    assert pd.api.types.is_integer_dtype(quantiles["paths"])


def test_build_quantiles_frame_defaults_quantiles_when_none_or_empty() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    default_quantiles = build_quantiles_frame(path_frame, None)
    empty_quantiles = build_quantiles_frame(path_frame, [])

    expected = [0.05, 0.5, 0.95]
    assert sorted(default_quantiles["quantile"].unique()) == pytest.approx(expected)
    assert sorted(empty_quantiles["quantile"].unique()) == pytest.approx(expected)


def test_build_quantiles_frame_accepts_scalar_quantile() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    quantiles = build_quantiles_frame(path_frame, 0.5)

    assert sorted(quantiles["quantile"].unique()) == pytest.approx([0.5])
    metric_row = quantiles.loc[quantiles["metric"] == "metric"].iloc[0]
    assert metric_row["value"] == pytest.approx(3.0)


def test_build_quantiles_frame_accepts_percent_string_quantiles() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    quantiles = build_quantiles_frame(path_frame, "5%, 50%, 95%")

    assert sorted(quantiles["quantile"].unique()) == pytest.approx([0.05, 0.5, 0.95])
    metric_row = quantiles.loc[
        (quantiles["metric"] == "metric") & (quantiles["quantile"] == 0.5)
    ].iloc[0]
    assert metric_row["value"] == pytest.approx(3.0)


def test_build_quantiles_frame_defaults_on_empty_string_quantiles() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    quantiles = build_quantiles_frame(path_frame, "  ")

    assert sorted(quantiles["quantile"].unique()) == pytest.approx([0.05, 0.5, 0.95])


def test_build_quantiles_frame_reports_all_metrics() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    quantiles = build_quantiles_frame(path_frame, [0.25, 0.75])

    assert set(quantiles["metric"]) == {"metric", "metric2"}
    assert sorted(quantiles["quantile"].unique()) == pytest.approx([0.25, 0.75])
    metric_rows = quantiles[quantiles["metric"] == "metric"]
    metric2_rows = quantiles[quantiles["metric"] == "metric2"]
    assert len(metric_rows) == 2
    assert len(metric2_rows) == 2
    assert set(metric_rows["paths"]) == {3}
    assert set(metric2_rows["paths"]) == {3}


def test_build_quantiles_frame_groups_by_strategy_and_fold() -> None:
    path_frame = build_path_frame(
        pd.DataFrame(
            [
                {"strategy": "A", "fold": 0, "path": 1, "metric": 1.0},
                {"strategy": "A", "fold": 0, "path": 2, "metric": 3.0},
                {"strategy": "A", "fold": 1, "path": 3, "metric": 2.0},
                {"strategy": "A", "fold": 1, "path": 4, "metric": 4.0},
                {"strategy": "B", "fold": 0, "path": 5, "metric": 10.0},
                {"strategy": "B", "fold": 0, "path": 6, "metric": 12.0},
            ]
        )
    )

    quantiles = build_quantiles_frame(path_frame, [0.5])

    assert len(quantiles) == 3
    assert quantiles.loc[(quantiles["strategy"] == "A") & (quantiles["fold"] == 0), "value"].iloc[
        0
    ] == pytest.approx(2.0)
    assert quantiles.loc[(quantiles["strategy"] == "A") & (quantiles["fold"] == 1), "value"].iloc[
        0
    ] == pytest.approx(3.0)
    assert quantiles.loc[(quantiles["strategy"] == "B") & (quantiles["fold"] == 0), "value"].iloc[
        0
    ] == pytest.approx(11.0)


def test_build_quantiles_frame_ignores_non_finite_values() -> None:
    path_frame = build_path_frame(
        pd.DataFrame(
            [
                {"strategy": "A", "path": 1, "fold": 0, "metric": 1.0},
                {"strategy": "A", "path": 2, "fold": 0, "metric": float("nan")},
                {"strategy": "A", "path": 3, "fold": 0, "metric": float("inf")},
                {"strategy": "A", "path": 4, "fold": 0, "metric": float("-inf")},
                {"strategy": "A", "path": 5, "fold": 0, "metric": 3.0},
            ]
        )
    )

    quantiles = build_quantiles_frame(path_frame, [0.5])

    assert quantiles.loc[0, "value"] == pytest.approx(2.0)
    assert quantiles.loc[0, "paths"] == 2


def test_build_quantiles_frame_reports_zero_paths_for_all_non_finite_metric() -> None:
    path_frame = build_path_frame(
        pd.DataFrame(
            [
                {"strategy": "A", "path": 1, "fold": 0, "metric": 1.0, "metric2": float("nan")},
                {"strategy": "A", "path": 2, "fold": 0, "metric": 2.0, "metric2": float("inf")},
                {"strategy": "A", "path": 3, "fold": 0, "metric": 3.0, "metric2": float("-inf")},
            ]
        )
    )

    quantiles = build_quantiles_frame(path_frame, [0.5])

    metric_row = quantiles.loc[quantiles["metric"] == "metric"].iloc[0]
    metric2_row = quantiles.loc[quantiles["metric"] == "metric2"].iloc[0]
    assert metric_row["paths"] == 3
    assert metric_row["value"] == pytest.approx(2.0)
    assert metric2_row["paths"] == 0
    assert pd.isna(metric2_row["value"])


def test_build_quantiles_frame_all_non_finite_metrics_report_zero_paths() -> None:
    path_frame = build_path_frame(
        pd.DataFrame(
            [
                {
                    "strategy": "A",
                    "path": 1,
                    "fold": 0,
                    "metric": float("nan"),
                    "metric2": float("inf"),
                },
                {
                    "strategy": "A",
                    "path": 2,
                    "fold": 0,
                    "metric": float("-inf"),
                    "metric2": float("nan"),
                },
            ]
        )
    )

    quantiles = build_quantiles_frame(path_frame, [0.5])

    assert set(quantiles["metric"]) == {"metric", "metric2"}
    assert quantiles["paths"].nunique() == 1
    assert quantiles["paths"].iloc[0] == 0
    assert quantiles["value"].isna().all()


def test_build_quantiles_frame_handles_string_metrics() -> None:
    path_frame = build_path_frame(
        pd.DataFrame(
            [
                {"strategy": "A", "path": 1, "fold": 0, "metric": "1.0"},
                {"strategy": "A", "path": 2, "fold": 0, "metric": "3.0"},
                {"strategy": "A", "path": 3, "fold": 0, "metric": "5.0"},
            ]
        )
    )

    quantiles = build_quantiles_frame(path_frame, [0.5])

    metric_row = quantiles.loc[quantiles["metric"] == "metric"].iloc[0]
    assert metric_row["value"] == pytest.approx(3.0)
    assert metric_row["paths"] == 3


def test_build_quantiles_frame_skips_non_finite_quantiles() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    quantiles = build_quantiles_frame(path_frame, [float("nan"), 0.5])

    assert len(quantiles) == 2
    assert sorted(quantiles["quantile"].unique()) == pytest.approx([0.5])


def test_build_quantiles_frame_rejects_out_of_bounds_quantile() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    with pytest.raises(ValueError, match="Quantiles must be between 0 and 1"):
        build_quantiles_frame(path_frame, [-0.1])


def test_build_quantiles_frame_rejects_bool_in_sequence() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    with pytest.raises(TypeError, match="Quantiles must be numeric values"):
        build_quantiles_frame(path_frame, [True, 0.5])


def test_build_quantiles_frame_defaults_when_quantiles_empty() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    quantiles = build_quantiles_frame(path_frame, [])

    assert sorted(quantiles["quantile"].unique()) == pytest.approx([0.05, 0.5, 0.95])


def test_build_quantiles_frame_defaults_when_quantiles_none() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    quantiles = build_quantiles_frame(path_frame, None)

    assert sorted(quantiles["quantile"].unique()) == pytest.approx([0.05, 0.5, 0.95])


def test_build_quantiles_frame_deduplicates_quantiles() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    quantiles = build_quantiles_frame(path_frame, [0.5, 0.5, 0.95, 0.5])

    assert sorted(quantiles["quantile"].unique()) == pytest.approx([0.5, 0.95])
    assert len(quantiles) == 4


def test_build_quantiles_frame_empty_input_preserves_schema() -> None:
    path_frame = pd.DataFrame(columns=list(AGGREGATION_PATH_COLUMNS))

    quantiles = build_quantiles_frame(path_frame, [0.5])

    assert quantiles.empty
    assert list(quantiles.columns) == list(quantiles_frame_schema())


def test_summary_frames_fill_missing_path_columns() -> None:
    path_frame = pd.DataFrame({"metric": [1.0, 2.0]})

    quantiles = build_quantiles_frame(path_frame, [0.5])
    breach = build_breach_frame(path_frame, {"metric": [1.5]})
    shortfall = build_expected_shortfall_frame(path_frame, {"metric": 0.5})

    assert quantiles["strategy"].isna().all()
    assert quantiles["fold"].isna().all()
    assert breach["strategy"].isna().all()
    assert breach["fold"].isna().all()
    assert shortfall["strategy"].isna().all()
    assert shortfall["fold"].isna().all()


def test_build_quantiles_frame_counts_paths_with_missing_strategy_fold() -> None:
    path_frame = pd.DataFrame({"metric": [1.0, 2.0, 3.0]})

    quantiles = build_quantiles_frame(path_frame, [0.5])

    assert quantiles.loc[0, "paths"] == 3
    assert quantiles.loc[0, "value"] == pytest.approx(2.0)


def test_build_breach_and_expected_shortfall_support_upper_tail() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    breach = build_breach_frame(
        path_frame,
        {"metric": {"thresholds": [2.5], "direction": "upper"}},
    )
    shortfall = build_expected_shortfall_frame(
        path_frame,
        {"metric": {"alpha": 0.5, "tail": "upper"}},
    )

    assert breach.loc[0, "direction"] == "upper"
    assert breach.loc[0, "breach_probability"] == pytest.approx(2.0 / 3.0)
    assert shortfall.loc[0, "tail"] == "upper"
    assert shortfall.loc[0, "threshold"] == pytest.approx(3.0)
    assert shortfall.loc[0, "expected_shortfall"] == pytest.approx(4.0)


def test_build_breach_frame_accepts_tail_alias() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    breach = build_breach_frame(
        path_frame,
        {"metric": {"thresholds": [2.5], "tail": "upper"}},
    )

    assert breach.loc[0, "direction"] == "upper"
    assert breach.loc[0, "breach_probability"] == pytest.approx(2.0 / 3.0)


def test_build_breach_frame_rejects_invalid_direction() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    with pytest.raises(ValueError, match="Unsupported breach direction"):
        build_breach_frame(
            path_frame,
            {"metric": {"thresholds": [1.0], "direction": "sideways"}},
        )


def test_build_breach_frame_defaults_direction_when_none() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    breach = build_breach_frame(
        path_frame,
        {"metric": {"thresholds": [2.5], "direction": None}},
    )

    assert breach.loc[0, "direction"] == "lower"
    assert breach.loc[0, "breach_probability"] == pytest.approx(1.0 / 3.0)


def test_build_breach_frame_supports_default_threshold_mapping() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    breach = build_breach_frame(
        path_frame,
        {"thresholds": [2.5], "direction": "upper"},
    )

    assert set(breach["metric"]) == {"metric", "metric2"}
    assert set(breach["direction"]) == {"upper"}
    metric_prob = breach.loc[breach["metric"] == "metric", "breach_probability"].iloc[0]
    metric2_prob = breach.loc[breach["metric"] == "metric2", "breach_probability"].iloc[0]
    assert metric_prob == pytest.approx(2.0 / 3.0)
    assert metric2_prob == pytest.approx(2.0 / 3.0)


def test_build_breach_frame_supports_default_key() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    breach = build_breach_frame(
        path_frame,
        {
            "metric": {"thresholds": [2.5], "direction": "lower"},
            "default": {"thresholds": [4.0], "direction": "lower"},
        },
    )

    metric_prob = breach.loc[breach["metric"] == "metric", "breach_probability"].iloc[0]
    metric2_prob = breach.loc[breach["metric"] == "metric2", "breach_probability"].iloc[0]
    assert metric_prob == pytest.approx(1.0 / 3.0)
    assert metric2_prob == pytest.approx(2.0 / 3.0)


def test_aggregate_monte_carlo_results_respects_quantile_config() -> None:
    results_frame = _sample_results_frame()

    aggregation = aggregate_monte_carlo_results(
        results_frame,
        quantiles=[0.25, 0.75],
        breach_spec={"metric": [2.5]},
        expected_shortfall_spec={"metric": 0.5},
    )

    assert list(aggregation.path_frame.columns) == list(AGGREGATION_PATH_COLUMNS) + [
        "metric",
        "metric2",
    ]
    quantile_values = sorted(aggregation.quantiles_frame["quantile"].unique())
    assert quantile_values == pytest.approx([0.25, 0.75])
    assert len(aggregation.quantiles_frame) == 4
    assert not aggregation.breach_frame.empty
    assert not aggregation.expected_shortfall_frame.empty


def test_aggregate_monte_carlo_results_accepts_percent_string_quantiles() -> None:
    results_frame = _sample_results_frame()

    aggregation = aggregate_monte_carlo_results(results_frame, quantiles="25%, 75%")

    quantile_values = sorted(aggregation.quantiles_frame["quantile"].unique())
    assert quantile_values == pytest.approx([0.25, 0.75])


def test_aggregate_monte_carlo_results_reports_expected_shortfall_values() -> None:
    results_frame = _sample_results_frame()

    aggregation = aggregate_monte_carlo_results(
        results_frame,
        quantiles=[0.5],
        breach_spec={"metric": [2.5]},
        expected_shortfall_spec={"metric": {"alpha": 0.5, "tail": "lower"}},
    )

    shortfall_row = aggregation.expected_shortfall_frame.loc[
        aggregation.expected_shortfall_frame["metric"] == "metric"
    ].iloc[0]

    assert shortfall_row["threshold"] == pytest.approx(3.0)
    assert shortfall_row["expected_shortfall"] == pytest.approx(2.0)


def test_aggregate_monte_carlo_results_empty_input_preserves_schemas() -> None:
    results_frame = pd.DataFrame(
        {
            "strategy": pd.Series(dtype=str),
            "path": pd.Series(dtype=int),
            "fold": pd.Series(dtype=int),
            "metric": pd.Series(dtype=float),
        }
    )

    aggregation = aggregate_monte_carlo_results(results_frame, quantiles=[0.5])

    assert aggregation.path_frame.empty
    assert list(aggregation.path_frame.columns) == list(AGGREGATION_PATH_COLUMNS) + ["metric"]
    assert aggregation.quantiles_frame.empty
    assert list(aggregation.quantiles_frame.columns) == list(quantiles_frame_schema())
    assert aggregation.breach_frame.empty
    assert list(aggregation.breach_frame.columns) == list(breach_frame_schema())
    assert aggregation.expected_shortfall_frame.empty
    assert list(aggregation.expected_shortfall_frame.columns) == list(
        expected_shortfall_frame_schema()
    )


def test_build_breach_frame_handles_lower_and_upper_thresholds() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    breach = build_breach_frame(
        path_frame,
        {
            "metric": {"thresholds": [2.5], "direction": "lower"},
            "metric2": {"thresholds": [5.0], "direction": "upper"},
        },
    )

    assert list(breach.columns) == list(breach_frame_schema())
    metric_prob = breach.loc[breach["metric"] == "metric", "breach_probability"].iloc[0]
    metric2_prob = breach.loc[breach["metric"] == "metric2", "breach_probability"].iloc[0]
    assert metric_prob == pytest.approx(1.0 / 3.0)
    assert metric2_prob == pytest.approx(1.0 / 3.0)


def test_build_breach_frame_groups_by_strategy_and_fold() -> None:
    path_frame = build_path_frame(
        pd.DataFrame(
            [
                {"strategy": "A", "fold": 0, "path": 1, "metric": 1.0},
                {"strategy": "A", "fold": 0, "path": 2, "metric": 3.0},
                {"strategy": "A", "fold": 1, "path": 3, "metric": 2.0},
                {"strategy": "A", "fold": 1, "path": 4, "metric": 4.0},
                {"strategy": "B", "fold": 0, "path": 5, "metric": 0.0},
                {"strategy": "B", "fold": 0, "path": 6, "metric": 2.0},
                {"strategy": "B", "fold": 1, "path": 7, "metric": 5.0},
                {"strategy": "B", "fold": 1, "path": 8, "metric": 6.0},
            ]
        )
    )

    breach = build_breach_frame(
        path_frame,
        {"metric": {"thresholds": [2.0], "direction": "lower"}},
    )

    assert len(breach) == 4
    for _, row in breach.iterrows():
        assert row["paths"] == 2
    assert breach.loc[
        (breach["strategy"] == "A") & (breach["fold"] == 0), "breach_probability"
    ].iloc[0] == pytest.approx(0.5)
    assert breach.loc[
        (breach["strategy"] == "A") & (breach["fold"] == 1), "breach_probability"
    ].iloc[0] == pytest.approx(0.5)
    assert breach.loc[
        (breach["strategy"] == "B") & (breach["fold"] == 0), "breach_probability"
    ].iloc[0] == pytest.approx(1.0)
    assert breach.loc[
        (breach["strategy"] == "B") & (breach["fold"] == 1), "breach_probability"
    ].iloc[0] == pytest.approx(0.0)


def test_build_breach_frame_accepts_threshold_key() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    breach = build_breach_frame(
        path_frame,
        {"metric": {"threshold": 2.5, "direction": "upper"}},
    )

    assert breach.loc[0, "threshold"] == pytest.approx(2.5)
    assert breach.loc[0, "direction"] == "upper"
    assert breach.loc[0, "breach_probability"] == pytest.approx(2.0 / 3.0)


def test_build_breach_frame_accepts_scalar_threshold() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    breach = build_breach_frame(path_frame, 2.5)

    assert list(breach.columns) == list(breach_frame_schema())
    assert len(breach) == 2
    assert breach["threshold"].nunique() == 1
    assert breach["threshold"].iloc[0] == pytest.approx(2.5)
    assert set(breach["metric"]) == {"metric", "metric2"}


def test_build_breach_frame_applies_default_thresholds_to_all_metrics() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    breach = build_breach_frame(path_frame, [4.0])

    assert list(breach.columns) == list(breach_frame_schema())
    assert len(breach) == 2
    assert set(breach["direction"].unique()) == {"lower"}
    metric_prob = breach.loc[breach["metric"] == "metric", "breach_probability"].iloc[0]
    metric2_prob = breach.loc[breach["metric"] == "metric2", "breach_probability"].iloc[0]
    assert metric_prob == pytest.approx(2.0 / 3.0)
    assert metric2_prob == pytest.approx(2.0 / 3.0)


def test_build_breach_frame_skips_non_finite_thresholds() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    breach = build_breach_frame(path_frame, {"metric": [float("nan"), 2.5]})

    assert len(breach) == 1
    assert breach.loc[0, "threshold"] == pytest.approx(2.5)


def test_build_breach_frame_skips_non_numeric_thresholds() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    breach = build_breach_frame(path_frame, {"metric": ["nope", 2.5]})

    assert len(breach) == 1
    assert breach.loc[0, "threshold"] == pytest.approx(2.5)


def test_build_breach_frame_skips_none_thresholds() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    breach = build_breach_frame(path_frame, {"metric": [None, 2.5]})

    assert len(breach) == 1
    assert breach.loc[0, "threshold"] == pytest.approx(2.5)


def test_build_breach_frame_skips_none_default_thresholds() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    breach = build_breach_frame(path_frame, [None, 4.0])

    assert len(breach) == 2
    assert breach["threshold"].nunique() == 1
    assert breach["threshold"].iloc[0] == pytest.approx(4.0)


def test_build_breach_frame_dedupes_thresholds() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    breach = build_breach_frame(path_frame, {"metric": [2.5, 2.5, 3.5, 2.5]})

    assert len(breach) == 2
    assert breach["threshold"].tolist() == pytest.approx([2.5, 3.5])


def test_build_breach_frame_empty_threshold_list_preserves_schema() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    breach = build_breach_frame(path_frame, [])

    assert breach.empty
    assert list(breach.columns) == list(breach_frame_schema())


def test_build_breach_frame_none_spec_preserves_schema() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    breach = build_breach_frame(path_frame, None)

    assert breach.empty
    assert list(breach.columns) == list(breach_frame_schema())


def test_build_breach_frame_ignores_unknown_metrics() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    breach = build_breach_frame(path_frame, {"unknown_metric": [1.0]})

    assert breach.empty
    assert list(breach.columns) == list(breach_frame_schema())


def test_build_breach_frame_ignores_non_finite_metric_values() -> None:
    path_frame = build_path_frame(
        pd.DataFrame(
            [
                {"strategy": "A", "path": 1, "fold": 0, "metric": 1.0},
                {"strategy": "A", "path": 2, "fold": 0, "metric": float("nan")},
                {"strategy": "A", "path": 3, "fold": 0, "metric": float("inf")},
                {"strategy": "A", "path": 4, "fold": 0, "metric": float("-inf")},
                {"strategy": "A", "path": 5, "fold": 0, "metric": 2.0},
            ]
        )
    )

    breach = build_breach_frame(path_frame, {"metric": {"thresholds": [1.5], "direction": "lower"}})

    assert breach.loc[0, "paths"] == 2
    assert breach.loc[0, "breach_probability"] == pytest.approx(0.5)


def test_build_breach_frame_ignores_non_numeric_metrics() -> None:
    results_frame = pd.DataFrame(
        [
            {"strategy": "A", "path": 1, "fold": 0, "metric": 1.0, "note": "x"},
            {"strategy": "A", "path": 2, "fold": 0, "metric": 2.0, "note": "y"},
        ]
    )
    path_frame = build_path_frame(results_frame)

    breach = build_breach_frame(path_frame, [1.5])

    assert set(breach["metric"]) == {"metric"}


def test_build_expected_shortfall_frame_computes_tail_mean() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    shortfall = build_expected_shortfall_frame(
        path_frame,
        {
            "metric": {"alpha": 0.5, "tail": "lower"},
            "metric2": {"alpha": 0.5, "tail": "upper"},
        },
    )

    assert list(shortfall.columns) == list(expected_shortfall_frame_schema())
    metric_es = shortfall.loc[shortfall["metric"] == "metric", "expected_shortfall"].iloc[0]
    metric2_es = shortfall.loc[shortfall["metric"] == "metric2", "expected_shortfall"].iloc[0]
    assert metric_es == pytest.approx(2.0)
    assert metric2_es == pytest.approx(5.0)


def test_build_expected_shortfall_frame_rejects_invalid_alpha() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    with pytest.raises(ValueError, match="Expected shortfall alpha must be between 0 and 1"):
        build_expected_shortfall_frame(
            path_frame,
            {"metric": {"alpha": 1.5, "tail": "lower"}},
        )


def test_build_expected_shortfall_reports_thresholds() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    shortfall = build_expected_shortfall_frame(
        path_frame,
        {
            "metric": {"alpha": 0.5, "tail": "lower"},
            "metric2": {"alpha": 0.5, "tail": "upper"},
        },
    )

    metric_threshold = shortfall.loc[shortfall["metric"] == "metric", "threshold"].iloc[0]
    metric2_threshold = shortfall.loc[shortfall["metric"] == "metric2", "threshold"].iloc[0]
    assert metric_threshold == pytest.approx(3.0)
    assert metric2_threshold == pytest.approx(4.0)


def test_build_expected_shortfall_accepts_direction_alias() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    shortfall = build_expected_shortfall_frame(
        path_frame,
        {"metric": {"alpha": 0.5, "direction": "upper"}},
    )

    metric_row = shortfall.loc[shortfall["metric"] == "metric"].iloc[0]
    assert metric_row["tail"] == "upper"
    assert metric_row["threshold"] == pytest.approx(3.0)
    assert metric_row["expected_shortfall"] == pytest.approx(4.0)


def test_build_expected_shortfall_defaults_tail_when_none() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    shortfall = build_expected_shortfall_frame(
        path_frame,
        {"metric": {"alpha": 0.5, "tail": None}},
    )

    metric_row = shortfall.loc[shortfall["metric"] == "metric"].iloc[0]
    assert metric_row["tail"] == "lower"
    assert metric_row["threshold"] == pytest.approx(3.0)
    assert metric_row["expected_shortfall"] == pytest.approx(2.0)


def test_build_expected_shortfall_defaults_to_all_metrics() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    shortfall = build_expected_shortfall_frame(path_frame, None)

    assert set(shortfall["metric"]) == {"metric", "metric2"}
    assert set(shortfall["tail"]) == {"lower"}
    assert shortfall["alpha"].tolist() == pytest.approx([0.05, 0.05])


def test_build_expected_shortfall_supports_default_mapping() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    shortfall = build_expected_shortfall_frame(
        path_frame,
        {"default": {"alpha": 0.5, "tail": "upper"}},
    )

    assert set(shortfall["metric"]) == {"metric", "metric2"}
    assert set(shortfall["tail"]) == {"upper"}
    assert shortfall["alpha"].tolist() == pytest.approx([0.5, 0.5])
    metric_es = shortfall.loc[shortfall["metric"] == "metric", "expected_shortfall"].iloc[0]
    metric2_es = shortfall.loc[shortfall["metric"] == "metric2", "expected_shortfall"].iloc[0]
    assert metric_es == pytest.approx(4.0)
    assert metric2_es == pytest.approx(5.0)


def test_build_expected_shortfall_supports_top_level_defaults() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    shortfall = build_expected_shortfall_frame(
        path_frame,
        {"alpha": 0.5, "tail": "upper"},
    )

    assert set(shortfall["metric"]) == {"metric", "metric2"}
    assert set(shortfall["tail"]) == {"upper"}
    assert shortfall["alpha"].tolist() == pytest.approx([0.5, 0.5])
    metric_es = shortfall.loc[shortfall["metric"] == "metric", "expected_shortfall"].iloc[0]
    metric2_es = shortfall.loc[shortfall["metric"] == "metric2", "expected_shortfall"].iloc[0]
    assert metric_es == pytest.approx(4.0)
    assert metric2_es == pytest.approx(5.0)


def test_build_expected_shortfall_accepts_scalar_alpha() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    shortfall = build_expected_shortfall_frame(path_frame, 0.2)

    assert set(shortfall["metric"]) == {"metric", "metric2"}
    assert set(shortfall["tail"]) == {"lower"}
    assert shortfall["alpha"].tolist() == pytest.approx([0.2, 0.2])


def test_build_expected_shortfall_defaults_when_spec_empty() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    shortfall = build_expected_shortfall_frame(path_frame, {})

    assert set(shortfall["metric"]) == {"metric", "metric2"}
    assert set(shortfall["tail"]) == {"lower"}
    assert shortfall["alpha"].tolist() == pytest.approx([0.05, 0.05])


def test_build_expected_shortfall_skips_none_spec() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    shortfall = build_expected_shortfall_frame(path_frame, {"metric": None})

    assert set(shortfall["metric"]) == {"metric", "metric2"}
    assert set(shortfall["tail"]) == {"lower"}
    assert shortfall["alpha"].tolist() == pytest.approx([0.05, 0.05])


def test_build_expected_shortfall_ignores_non_finite_values() -> None:
    path_frame = build_path_frame(
        pd.DataFrame(
            [
                {"strategy": "A", "path": 1, "fold": 0, "metric": 1.0},
                {"strategy": "A", "path": 2, "fold": 0, "metric": float("nan")},
                {"strategy": "A", "path": 3, "fold": 0, "metric": float("inf")},
                {"strategy": "A", "path": 4, "fold": 0, "metric": float("-inf")},
                {"strategy": "A", "path": 5, "fold": 0, "metric": 5.0},
            ]
        )
    )

    shortfall = build_expected_shortfall_frame(path_frame, {"metric": {"alpha": 0.5}})

    assert shortfall.loc[0, "paths"] == 2
    assert shortfall.loc[0, "expected_shortfall"] == pytest.approx(1.0)


def test_build_expected_shortfall_handles_all_non_finite_values() -> None:
    path_frame = build_path_frame(
        pd.DataFrame(
            [
                {"strategy": "A", "path": 1, "fold": 0, "metric": float("nan")},
                {"strategy": "A", "path": 2, "fold": 0, "metric": float("inf")},
                {"strategy": "A", "path": 3, "fold": 0, "metric": float("-inf")},
            ]
        )
    )

    shortfall = build_expected_shortfall_frame(path_frame, {"metric": {"alpha": 0.5}})

    assert shortfall.loc[0, "paths"] == 0
    assert pd.isna(shortfall.loc[0, "threshold"])
    assert pd.isna(shortfall.loc[0, "expected_shortfall"])


def test_build_expected_shortfall_ignores_non_numeric_metrics() -> None:
    results_frame = pd.DataFrame(
        [
            {"strategy": "A", "path": 1, "fold": 0, "metric": 1.0, "note": "x"},
            {"strategy": "A", "path": 2, "fold": 0, "metric": 2.0, "note": "y"},
        ]
    )
    path_frame = build_path_frame(results_frame)

    shortfall = build_expected_shortfall_frame(path_frame, None)

    assert set(shortfall["metric"]) == {"metric"}


def test_build_expected_shortfall_rejects_non_finite_alpha() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    with pytest.raises(ValueError, match="Expected shortfall alpha must be between 0 and 1"):
        build_expected_shortfall_frame(path_frame, {"metric": {"alpha": float("nan")}})


def test_build_expected_shortfall_rejects_boolean_alpha() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    with pytest.raises(TypeError, match="Expected shortfall alpha must be numeric values"):
        build_expected_shortfall_frame(path_frame, True)

    with pytest.raises(TypeError, match="Expected shortfall alpha must be numeric values"):
        build_expected_shortfall_frame(path_frame, {"metric": {"alpha": True}})


def test_build_expected_shortfall_rejects_scalar_alpha_out_of_range() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    with pytest.raises(ValueError, match="Expected shortfall alpha must be between 0 and 1"):
        build_expected_shortfall_frame(path_frame, 0.0)


def test_build_expected_shortfall_rejects_alpha_at_bounds() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    with pytest.raises(ValueError, match="Expected shortfall alpha must be between 0 and 1"):
        build_expected_shortfall_frame(path_frame, {"metric": {"alpha": 0.0}})

    with pytest.raises(ValueError, match="Expected shortfall alpha must be between 0 and 1"):
        build_expected_shortfall_frame(path_frame, {"metric": {"alpha": 1.0}})


def test_build_expected_shortfall_empty_input_preserves_schema() -> None:
    path_frame = pd.DataFrame(columns=list(AGGREGATION_PATH_COLUMNS))

    shortfall = build_expected_shortfall_frame(path_frame, None)

    assert shortfall.empty
    assert list(shortfall.columns) == list(expected_shortfall_frame_schema())


def test_schema_helpers_match_column_constants() -> None:
    assert PATH_COLUMNS == AGGREGATION_PATH_COLUMNS
    assert quantiles_frame_schema() == QUANTILE_COLUMNS
    assert breach_frame_schema() == BREACH_COLUMNS
    assert expected_shortfall_frame_schema() == EXPECTED_SHORTFALL_COLUMNS


def test_aggregation_frame_schemas_reports_all_outputs() -> None:
    results_frame = _sample_results_frame()

    schemas = aggregation_frame_schemas(results_frame)

    assert schemas["path"] == tuple(AGGREGATION_PATH_COLUMNS) + ("metric", "metric2")
    assert schemas["quantiles"] == QUANTILE_COLUMNS
    assert schemas["breach"] == BREACH_COLUMNS
    assert schemas["expected_shortfall"] == EXPECTED_SHORTFALL_COLUMNS


def test_aggregation_frame_schemas_empty_input_includes_metric_columns() -> None:
    results_frame = pd.DataFrame(
        {
            "strategy": pd.Series(dtype=str),
            "path": pd.Series(dtype=int),
            "fold": pd.Series(dtype=int),
            "metric": pd.Series(dtype=float),
        }
    )

    schemas = aggregation_frame_schemas(results_frame)

    assert schemas["path"] == tuple(AGGREGATION_PATH_COLUMNS) + ("metric",)
    assert schemas["quantiles"] == QUANTILE_COLUMNS
    assert schemas["breach"] == BREACH_COLUMNS
    assert schemas["expected_shortfall"] == EXPECTED_SHORTFALL_COLUMNS


def test_export_aggregation_results_writes_csv(tmp_path) -> None:
    results_frame = _sample_results_frame()
    path_frame = build_path_frame(results_frame)
    quantiles = build_quantiles_frame(path_frame, [0.5])
    breach = build_breach_frame(path_frame, {"metric": [2.5]})
    shortfall = build_expected_shortfall_frame(path_frame, {"metric": 0.5})

    aggregation = MonteCarloAggregationResults(
        path_frame=path_frame,
        quantiles_frame=quantiles,
        breach_frame=breach,
        expected_shortfall_frame=shortfall,
    )

    exported = export_aggregation_results(aggregation, tmp_path, formats=["csv"])

    assert exported["path_summary_csv"].exists()
    assert exported["per_strategy_stats_csv"].exists()
    assert exported["per_strategy_path_csv"].exists()
    assert exported["quantiles_csv"].exists()
    assert exported["summary_quantiles_csv"].exists()
    assert exported["breach_probabilities_csv"].exists()
    assert exported["expected_shortfall_csv"].exists()


def test_export_aggregation_results_defaults_to_csv_when_parquet_unavailable(
    tmp_path, monkeypatch
) -> None:
    results_frame = _sample_results_frame()
    aggregation = aggregate_monte_carlo_results(
        results_frame,
        quantiles=[0.5],
        breach_spec={"metric": [2.5]},
        expected_shortfall_spec={"metric": 0.5},
    )

    monkeypatch.setattr(export_module, "_supports_parquet", lambda: False)
    exported = export_aggregation_results(aggregation, tmp_path)

    assert exported["path_summary_csv"].exists()
    assert exported["breach_probabilities_csv"].exists()
    assert "path_summary_parquet" not in exported
    assert "breach_probabilities_parquet" not in exported


def test_export_aggregation_results_defaults_to_csv_and_parquet_when_available(
    tmp_path,
) -> None:
    if not export_module._supports_parquet():
        pytest.skip("Parquet engine not available")

    results_frame = _sample_results_frame()
    aggregation = aggregate_monte_carlo_results(
        results_frame,
        quantiles=[0.5],
        breach_spec={"metric": [2.5]},
        expected_shortfall_spec={"metric": 0.5},
    )

    exported = export_aggregation_results(aggregation, tmp_path)

    assert exported["path_summary_csv"].exists()
    assert exported["path_summary_parquet"].exists()


def test_export_aggregation_results_path_summary_columns(tmp_path) -> None:
    results_frame = _sample_results_frame()

    aggregation = aggregate_monte_carlo_results(
        results_frame,
        quantiles=[0.5],
        breach_spec={"metric": [2.5]},
        expected_shortfall_spec={"metric": 0.5},
    )

    exported = export_aggregation_results(aggregation, tmp_path, formats=["csv"])
    path_summary = pd.read_csv(exported["path_summary_csv"])

    assert list(path_summary.columns[: len(PATH_COLUMNS)]) == list(PATH_COLUMNS)
    assert set(path_summary.columns[len(PATH_COLUMNS) :]) == {"metric", "metric2"}


def test_export_aggregation_results_per_strategy_path_columns(tmp_path) -> None:
    results_frame = _sample_results_frame()

    aggregation = aggregate_monte_carlo_results(
        results_frame,
        quantiles=[0.5],
        breach_spec={"metric": [2.5]},
        expected_shortfall_spec={"metric": 0.5},
    )

    exported = export_aggregation_results(aggregation, tmp_path, formats=["csv"])
    per_strategy_path = pd.read_csv(exported["per_strategy_path_csv"])

    assert list(per_strategy_path.columns[: len(PATH_COLUMNS)]) == list(PATH_COLUMNS)
    assert set(per_strategy_path.columns[len(PATH_COLUMNS) :]) == {"metric", "metric2"}


def test_export_aggregation_results_per_strategy_aliases_path_summary(tmp_path) -> None:
    results_frame = _sample_results_frame()

    aggregation = aggregate_monte_carlo_results(
        results_frame,
        quantiles=[0.5],
        breach_spec={"metric": [2.5]},
        expected_shortfall_spec={"metric": 0.5},
    )

    exported = export_aggregation_results(aggregation, tmp_path, formats=["csv"])
    path_summary = pd.read_csv(exported["path_summary_csv"])
    per_strategy_path = pd.read_csv(exported["per_strategy_path_csv"])
    per_strategy_stats = pd.read_csv(exported["per_strategy_stats_csv"])

    assert list(per_strategy_path.columns) == list(path_summary.columns)
    assert list(per_strategy_stats.columns) == list(path_summary.columns)
    assert len(per_strategy_path) == len(path_summary)
    assert len(per_strategy_stats) == len(path_summary)


def test_export_aggregation_results_writes_parquet(tmp_path) -> None:
    results_frame = _sample_results_frame()
    path_frame = build_path_frame(results_frame)
    quantiles = build_quantiles_frame(path_frame, [0.5])
    breach = build_breach_frame(path_frame, {"metric": [2.5]})
    shortfall = build_expected_shortfall_frame(path_frame, {"metric": 0.5})

    aggregation = MonteCarloAggregationResults(
        path_frame=path_frame,
        quantiles_frame=quantiles,
        breach_frame=breach,
        expected_shortfall_frame=shortfall,
    )

    exported = export_aggregation_results(aggregation, tmp_path, formats=["parquet"])

    assert exported["path_summary_parquet"].exists()
    assert exported["per_strategy_stats_parquet"].exists()
    assert exported["per_strategy_path_parquet"].exists()
    assert exported["quantiles_parquet"].exists()
    assert exported["summary_quantiles_parquet"].exists()
    assert exported["breach_probabilities_parquet"].exists()
    assert exported["expected_shortfall_parquet"].exists()
    path_summary = pd.read_parquet(exported["path_summary_parquet"])
    assert list(path_summary.columns) == list(AGGREGATION_PATH_COLUMNS) + ["metric", "metric2"]


def test_export_aggregation_results_path_summary_schema(tmp_path) -> None:
    results_frame = _sample_results_frame()

    aggregation = aggregate_monte_carlo_results(
        results_frame,
        quantiles=[0.5],
        breach_spec={"metric": [2.5]},
        expected_shortfall_spec={"metric": 0.5},
    )

    exported = export_aggregation_results(aggregation, tmp_path, formats=["csv"])
    path_summary = pd.read_csv(exported["path_summary_csv"])

    assert list(path_summary.columns) == list(AGGREGATION_PATH_COLUMNS) + ["metric", "metric2"]


def test_export_aggregation_results_path_summary_schema_without_metrics(tmp_path) -> None:
    results_frame = pd.DataFrame(
        [
            {"strategy": "A", "path": 1, "fold": 0, "note": "x"},
            {"strategy": "A", "path": 2, "fold": 0, "note": "y"},
        ]
    )

    aggregation = aggregate_monte_carlo_results(
        results_frame,
        quantiles=[0.5],
        breach_spec={"metric": [2.5]},
        expected_shortfall_spec={"metric": 0.5},
    )

    exported = export_aggregation_results(aggregation, tmp_path, formats=["csv"])
    path_summary = pd.read_csv(exported["path_summary_csv"])

    assert list(path_summary.columns) == list(AGGREGATION_PATH_COLUMNS)


def test_export_aggregation_results_adds_missing_path_columns(tmp_path) -> None:
    path_frame = pd.DataFrame({"metric": [1.0, 2.0]})
    quantiles_frame = pd.DataFrame(columns=list(QUANTILE_COLUMNS))
    breach_frame = pd.DataFrame(columns=list(BREACH_COLUMNS))
    shortfall_frame = pd.DataFrame(columns=list(EXPECTED_SHORTFALL_COLUMNS))

    aggregation = MonteCarloAggregationResults(
        path_frame=path_frame,
        quantiles_frame=quantiles_frame,
        breach_frame=breach_frame,
        expected_shortfall_frame=shortfall_frame,
    )

    exported = export_aggregation_results(aggregation, tmp_path, formats=["csv"])
    path_summary = pd.read_csv(exported["path_summary_csv"])

    assert list(path_summary.columns) == list(AGGREGATION_PATH_COLUMNS) + ["metric"]


def test_export_aggregation_results_summary_schemas(tmp_path) -> None:
    results_frame = _sample_results_frame()

    aggregation = aggregate_monte_carlo_results(
        results_frame,
        quantiles=[0.5],
        breach_spec={"metric": [2.5]},
        expected_shortfall_spec={"metric": 0.5},
    )

    exported = export_aggregation_results(aggregation, tmp_path, formats=["csv"])
    quantiles = pd.read_csv(exported["quantiles_csv"])
    summary_quantiles = pd.read_csv(exported["summary_quantiles_csv"])
    breach = pd.read_csv(exported["breach_probabilities_csv"])
    shortfall = pd.read_csv(exported["expected_shortfall_csv"])

    assert list(quantiles.columns) == list(quantiles_frame_schema())
    assert list(summary_quantiles.columns) == ["strategy", "fold", "metric", "q50"]
    assert list(breach.columns) == list(breach_frame_schema())
    assert list(shortfall.columns) == list(expected_shortfall_frame_schema())


def test_export_summary_quantiles_columns_for_multiple_quantiles(tmp_path) -> None:
    results_frame = _sample_results_frame()

    aggregation = aggregate_monte_carlo_results(
        results_frame,
        quantiles=[0.05, 0.5, 0.95],
        breach_spec={"metric": [2.5]},
        expected_shortfall_spec={"metric": 0.5},
    )

    exported = export_aggregation_results(aggregation, tmp_path, formats=["csv"])
    summary_quantiles = pd.read_csv(exported["summary_quantiles_csv"])

    assert list(summary_quantiles.columns) == ["strategy", "fold", "metric", "q05", "q50", "q95"]


def test_export_summary_quantiles_columns_for_fractional_quantiles(tmp_path) -> None:
    results_frame = _sample_results_frame()

    aggregation = aggregate_monte_carlo_results(
        results_frame,
        quantiles=[0.125, 0.5],
        breach_spec={"metric": [2.5]},
        expected_shortfall_spec={"metric": 0.5},
    )

    exported = export_aggregation_results(aggregation, tmp_path, formats=["csv"])
    summary_quantiles = pd.read_csv(exported["summary_quantiles_csv"])

    assert list(summary_quantiles.columns) == ["strategy", "fold", "metric", "q12_5", "q50"]


def test_export_aggregation_results_reorders_empty_frames(tmp_path) -> None:
    path_frame = pd.DataFrame(columns=["metric", "fold", "path", "strategy"])
    quantiles_frame = pd.DataFrame(columns=list(QUANTILE_COLUMNS)[::-1])
    breach_frame = pd.DataFrame(columns=list(BREACH_COLUMNS)[::-1])
    shortfall_frame = pd.DataFrame(columns=list(EXPECTED_SHORTFALL_COLUMNS)[::-1])

    aggregation = MonteCarloAggregationResults(
        path_frame=path_frame,
        quantiles_frame=quantiles_frame,
        breach_frame=breach_frame,
        expected_shortfall_frame=shortfall_frame,
    )

    exported = export_aggregation_results(aggregation, tmp_path, formats=["csv"])

    path_summary = pd.read_csv(exported["path_summary_csv"])
    quantiles = pd.read_csv(exported["quantiles_csv"])
    breach = pd.read_csv(exported["breach_probabilities_csv"])
    shortfall = pd.read_csv(exported["expected_shortfall_csv"])

    assert list(path_summary.columns) == list(AGGREGATION_PATH_COLUMNS) + ["metric"]
    assert list(quantiles.columns) == list(QUANTILE_COLUMNS)
    assert list(breach.columns) == list(BREACH_COLUMNS)
    assert list(shortfall.columns) == list(EXPECTED_SHORTFALL_COLUMNS)


def test_export_aggregation_results_adds_missing_schema_columns(tmp_path) -> None:
    path_frame = pd.DataFrame(columns=list(AGGREGATION_PATH_COLUMNS))
    quantiles_frame = pd.DataFrame([{"strategy": "A", "metric": "metric"}])
    breach_frame = pd.DataFrame([{"strategy": "A"}])
    shortfall_frame = pd.DataFrame([{"metric": "metric"}])

    aggregation = MonteCarloAggregationResults(
        path_frame=path_frame,
        quantiles_frame=quantiles_frame,
        breach_frame=breach_frame,
        expected_shortfall_frame=shortfall_frame,
    )

    exported = export_aggregation_results(aggregation, tmp_path, formats=["csv"])

    quantiles = pd.read_csv(exported["quantiles_csv"])
    breach = pd.read_csv(exported["breach_probabilities_csv"])
    shortfall = pd.read_csv(exported["expected_shortfall_csv"])

    assert list(quantiles.columns) == list(QUANTILE_COLUMNS)
    assert list(breach.columns) == list(BREACH_COLUMNS)
    assert list(shortfall.columns) == list(EXPECTED_SHORTFALL_COLUMNS)


def test_export_aggregation_results_supports_parquet(tmp_path) -> None:
    pytest.importorskip("pyarrow")

    results_frame = _sample_results_frame()
    path_frame = build_path_frame(results_frame)
    quantiles = build_quantiles_frame(path_frame, [0.5])
    breach = build_breach_frame(path_frame, {"metric": [2.5]})
    shortfall = build_expected_shortfall_frame(path_frame, {"metric": 0.5})

    aggregation = MonteCarloAggregationResults(
        path_frame=path_frame,
        quantiles_frame=quantiles,
        breach_frame=breach,
        expected_shortfall_frame=shortfall,
    )

    exported = export_aggregation_results(aggregation, tmp_path, formats=["parquet"])

    assert exported["path_summary_parquet"].exists()
    assert exported["quantiles_parquet"].exists()
    assert exported["breach_probabilities_parquet"].exists()
    assert exported["expected_shortfall_parquet"].exists()


def test_export_aggregation_results_supports_csv_and_parquet(tmp_path) -> None:
    pytest.importorskip("pyarrow")

    results_frame = _sample_results_frame()
    aggregation = aggregate_monte_carlo_results(
        results_frame,
        quantiles=[0.5],
        breach_spec={"metric": [2.5]},
        expected_shortfall_spec={"metric": 0.5},
    )

    exported = export_aggregation_results(
        aggregation,
        tmp_path,
        formats=["csv", "parquet"],
    )

    assert exported["path_summary_csv"].exists()
    assert exported["per_strategy_stats_csv"].exists()
    assert exported["per_strategy_path_csv"].exists()
    assert exported["quantiles_csv"].exists()
    assert exported["summary_quantiles_csv"].exists()
    assert exported["breach_probabilities_csv"].exists()
    assert exported["expected_shortfall_csv"].exists()
    assert exported["path_summary_parquet"].exists()
    assert exported["per_strategy_stats_parquet"].exists()
    assert exported["per_strategy_path_parquet"].exists()
    assert exported["quantiles_parquet"].exists()
    assert exported["summary_quantiles_parquet"].exists()
    assert exported["breach_probabilities_parquet"].exists()
    assert exported["expected_shortfall_parquet"].exists()


def test_export_aggregation_results_supports_comma_separated_formats(tmp_path) -> None:
    pytest.importorskip("pyarrow")

    results_frame = _sample_results_frame()
    aggregation = aggregate_monte_carlo_results(
        results_frame,
        quantiles=[0.5],
        breach_spec={"metric": [2.5]},
        expected_shortfall_spec={"metric": 0.5},
    )

    exported = export_aggregation_results(
        aggregation,
        tmp_path,
        formats="csv, parquet",
    )

    assert exported["path_summary_csv"].exists()
    assert exported["path_summary_parquet"].exists()


def test_export_aggregation_results_dedupes_formats(tmp_path, monkeypatch) -> None:
    pytest.importorskip("pyarrow")

    results_frame = _sample_results_frame()
    aggregation = aggregate_monte_carlo_results(
        results_frame,
        quantiles=[0.5],
        breach_spec={"metric": [2.5]},
        expected_shortfall_spec={"metric": 0.5},
    )

    calls: list[str] = []

    original_export = export_module._export_frame

    def _spy_export_frame(frame: pd.DataFrame, path: export_module.Path, fmt: str) -> None:
        calls.append(fmt)
        original_export(frame, path, fmt)

    monkeypatch.setattr(export_module, "_export_frame", _spy_export_frame)

    exported = export_module.export_aggregation_results(
        aggregation,
        tmp_path,
        formats=["csv", "CSV", "csv", "parquet", "parquet"],
    )

    assert len(calls) == 14
    assert exported["path_summary_csv"].exists()
    assert exported["path_summary_parquet"].exists()
