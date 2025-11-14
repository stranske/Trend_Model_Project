"""Soft coverage tests for the walk-forward engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.engine import walkforward as walk_mod


def make_sample_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-31", periods=12, freq="M")
    return pd.DataFrame(
        {
            "Date": dates,
            "metric_a": np.linspace(0.01, 0.12, len(dates)),
            "metric_b": np.linspace(0.02, 0.24, len(dates)),
        }
    )


def test_prepare_index_accepts_date_column_and_validates() -> None:
    df = make_sample_frame()
    prepared = walk_mod._prepare_index(df)
    assert isinstance(prepared.index, pd.DatetimeIndex)

    with pytest.raises(ValueError):
        walk_mod._prepare_index(pd.DataFrame({"value": [1, 2, 3]}))


def test_generate_splits_creates_expected_windows() -> None:
    index = pd.date_range("2024-01-31", periods=6, freq="M")
    splits = walk_mod._generate_splits(index, train=3, test=2, step=1)
    assert len(splits) == 2
    assert splits[0].train_end == index[2]
    assert splits[0].test_end == index[4]


def test_to_dataframe_and_flatten_results() -> None:
    series = pd.Series([1.0, 2.0], index=["metric_a", "metric_b"], name="mean")
    df = walk_mod._to_dataframe(series)
    assert df.index.tolist() == ["mean"]

    flattened = walk_mod._flatten_agg_result(df)
    assert ("metric_a", "mean") in flattened.index

    with pytest.raises(TypeError):
        walk_mod._to_dataframe(123)  # type: ignore[arg-type]


def test_information_ratio_frame_handles_scalar_and_series() -> None:
    columns = pd.Index(["metric_a", "metric_b"])
    scalar = walk_mod._information_ratio_frame(0.5, columns)
    assert scalar.columns.tolist() == columns.tolist()

    series = walk_mod._information_ratio_frame(pd.Series({"metric_b": 0.7}), columns)
    assert series.loc[:, "metric_b"].notna().all()


def test_infer_periods_per_year_handles_edge_cases() -> None:
    index = pd.date_range("2024-01-31", periods=10, freq="D")
    assert walk_mod._infer_periods_per_year(index) >= 252
    assert walk_mod._infer_periods_per_year(pd.DatetimeIndex([])) == 1


def test_agg_label_prefers_callable_name() -> None:
    assert walk_mod._agg_label("mean") == "mean"

    def custom_func(values: pd.DataFrame) -> pd.Series:
        return values.mean()

    assert walk_mod._agg_label(custom_func) == "custom_func"


def test_walk_forward_runs_with_regimes_and_mapping() -> None:
    df = walk_mod._prepare_index(make_sample_frame())
    regimes = pd.Series(
        np.where(df.index.month % 2 == 0, "even", "odd"), index=df.index
    )
    result = walk_mod.walk_forward(
        df,
        train_size=4,
        test_size=2,
        step_size=2,
        metric_cols=["metric_a", "metric_b"],
        regimes=regimes,
        agg="mean",
    )

    assert result.oos_windows.shape[0] == len(result.splits)
    assert result.periods_per_year >= 12
