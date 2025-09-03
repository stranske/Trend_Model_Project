"""Edge-case tests for viz charts functions (Issue #428).

These tests ensure robustness around missing assets in weights history
and confirm that helper functions sort indices and fill missing values.
They also sanity-check that plotting functions return a Figure + DataFrame.
"""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure

from trend_analysis.viz import charts


def test_turnover_series_handles_missing_assets():
    """Turnover accounts for assets appearing/disappearing (filled as 0)."""
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    weights_map: dict[pd.Timestamp, pd.Series] = {
        dates[0]: pd.Series({"A": 0.6, "B": 0.4}),
        dates[1]: pd.Series({"A": 0.7}),  # B missing -> treated as 0
        dates[2]: pd.Series({"B": 0.5}),  # A missing -> treated as 0
    }

    result = charts.turnover_series(weights_map)
    assert isinstance(result, tuple)
    fig, df = result
    assert isinstance(fig, Figure)
    assert list(df.columns) == ["turnover"]
    assert len(df) == 3

    # Verify via explicit reconstruction of filled frame
    w_df = charts._weights_to_frame(weights_map)
    expected = w_df.diff().abs().sum(axis=1).to_frame("turnover")
    pd.testing.assert_frame_equal(df, expected)


def test_weights_heatmap_data_sorts_and_fills():
    """weights_heatmap_data sorts index and fills NaNs with 0.0."""
    dates = pd.to_datetime(["2020-03-31", "2020-01-31", "2020-02-29"])  # out of order
    weights_map = {
        dates[0]: pd.Series({"A": 0.2}),
        dates[1]: pd.Series({"A": 0.5, "B": 0.5}),
        dates[2]: pd.Series({"B": 1.0}),
    }

    df = charts.weights_heatmap_data(weights_map)
    # Sorted ascending
    expected_index = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    pd.testing.assert_index_equal(df.index, pd.Index(expected_index))
    # Filled with zeros for missing assets
    assert set(df.columns) == {"A", "B"}
    assert df.loc[pd.Timestamp("2020-02-29"), "A"] == 0.0  # no A on Feb 2020 input
    assert df.loc[pd.Timestamp("2020-03-31"), "B"] == 0.0  # no B on Mar 2020 input
