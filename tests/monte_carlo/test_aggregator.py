from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.monte_carlo.aggregator import (
    AGGREGATION_PATH_COLUMNS,
    BREACH_COLUMNS,
    EXPECTED_SHORTFALL_COLUMNS,
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


def test_build_quantiles_frame_reports_requested_quantiles() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    quantiles = build_quantiles_frame(path_frame, [0.5])

    assert list(quantiles.columns) == list(quantiles_frame_schema())
    assert quantiles.loc[0, "quantile"] == pytest.approx(0.5)
    assert quantiles.loc[0, "metric"] == "metric"
    assert quantiles.loc[0, "value"] == pytest.approx(3.0)
    assert quantiles.loc[0, "paths"] == 3


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


def test_build_quantiles_frame_defaults_when_quantiles_empty() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    quantiles = build_quantiles_frame(path_frame, [])

    assert sorted(quantiles["quantile"].unique()) == pytest.approx([0.05, 0.5, 0.95])


def test_build_quantiles_frame_empty_input_preserves_schema() -> None:
    path_frame = pd.DataFrame(columns=list(AGGREGATION_PATH_COLUMNS))

    quantiles = build_quantiles_frame(path_frame, [0.5])

    assert quantiles.empty
    assert list(quantiles.columns) == list(quantiles_frame_schema())


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


def test_build_breach_frame_rejects_invalid_direction() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    with pytest.raises(ValueError, match="Unsupported breach direction"):
        build_breach_frame(
            path_frame,
            {"metric": {"thresholds": [1.0], "direction": "sideways"}},
        )


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


def test_build_breach_frame_empty_threshold_list_preserves_schema() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    breach = build_breach_frame(path_frame, [])

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


def test_build_expected_shortfall_defaults_to_all_metrics() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    shortfall = build_expected_shortfall_frame(path_frame, None)

    assert set(shortfall["metric"]) == {"metric", "metric2"}
    assert set(shortfall["tail"]) == {"lower"}
    assert shortfall["alpha"].tolist() == pytest.approx([0.05, 0.05])


def test_build_expected_shortfall_defaults_when_spec_empty() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    shortfall = build_expected_shortfall_frame(path_frame, {})

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


def test_build_expected_shortfall_rejects_non_finite_alpha() -> None:
    path_frame = build_path_frame(_sample_results_frame())

    with pytest.raises(ValueError, match="Expected shortfall alpha must be between 0 and 1"):
        build_expected_shortfall_frame(path_frame, {"metric": {"alpha": float("nan")}})


def test_build_expected_shortfall_empty_input_preserves_schema() -> None:
    path_frame = pd.DataFrame(columns=list(AGGREGATION_PATH_COLUMNS))

    shortfall = build_expected_shortfall_frame(path_frame, None)

    assert shortfall.empty
    assert list(shortfall.columns) == list(expected_shortfall_frame_schema())


def test_schema_helpers_match_column_constants() -> None:
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
    assert exported["quantiles_csv"].exists()
    assert exported["breach_probabilities_csv"].exists()
    assert exported["expected_shortfall_csv"].exists()


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
    breach = pd.read_csv(exported["breach_probabilities_csv"])
    shortfall = pd.read_csv(exported["expected_shortfall_csv"])

    assert list(quantiles.columns) == list(quantiles_frame_schema())
    assert list(breach.columns) == list(breach_frame_schema())
    assert list(shortfall.columns) == list(expected_shortfall_frame_schema())


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
    assert exported["quantiles_csv"].exists()
    assert exported["breach_probabilities_csv"].exists()
    assert exported["expected_shortfall_csv"].exists()
    assert exported["path_summary_parquet"].exists()
    assert exported["quantiles_parquet"].exists()
    assert exported["breach_probabilities_parquet"].exists()
    assert exported["expected_shortfall_parquet"].exists()
