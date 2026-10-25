"""Tests for the shared Streamlit chart helpers in ``streamlit_app/components/charts.py``."""

from __future__ import annotations

import pandas as pd

from streamlit_app.components import charts


def _series() -> pd.Series:
    index = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    return pd.Series([1.0, 1.5, 1.25], index=index)


def test_series_to_frame_empty_and_none_return_a_labelled_empty_frame() -> None:
    for empty in (None, pd.Series(dtype=float)):
        frame = charts._series_to_frame(empty, "Equity")
        assert list(frame.columns) == ["Date", "Equity"]
        assert frame.empty


def test_series_to_frame_drops_rows_whose_date_cannot_be_parsed() -> None:
    series = pd.Series([1.0, 2.0, 3.0], index=["2024-01-31", "not a date", "2024-03-31"])
    frame = charts._series_to_frame(series, "Equity")
    assert list(frame.columns) == ["Date", "Equity"]
    assert frame["Equity"].tolist() == [1.0, 3.0]
    assert frame["Date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-31", "2024-03-31"]


def test_equity_chart_is_a_blue_line_of_height_280_over_the_series() -> None:
    spec = charts.equity_chart(_series()).to_dict()
    assert spec["mark"]["type"] == "line"
    assert spec["mark"]["color"] == charts.PALETTE[0]
    assert spec["mark"]["strokeWidth"] == 2
    assert spec["height"] == 280
    assert spec["encoding"]["y"]["field"] == "Equity"
    values = spec["datasets"][spec["data"]["name"]]
    assert [row["Equity"] for row in values] == [1.0, 1.5, 1.25]


def test_drawdown_chart_is_an_orange_area_with_percent_axis() -> None:
    spec = charts.drawdown_chart(_series() - 1.5).to_dict()
    assert spec["mark"]["type"] == "area"
    assert spec["mark"]["color"] == charts.PALETTE[1]
    assert spec["mark"]["opacity"] == 0.45
    assert spec["encoding"]["y"]["axis"]["format"] == ".0%"


def test_rolling_sharpe_chart_is_a_dashed_green_line() -> None:
    spec = charts.rolling_sharpe_chart(_series()).to_dict()
    assert spec["mark"]["type"] == "line"
    assert spec["mark"]["color"] == charts.PALETTE[2]
    assert spec["mark"]["strokeDash"] == [6, 4]
    assert spec["encoding"]["y"]["title"] == "Rolling Sharpe"


def test_turnover_chart_is_an_indigo_bar_titled_rebalance() -> None:
    spec = charts.turnover_chart(_series()).to_dict()
    assert spec["mark"]["type"] == "bar"
    assert spec["mark"]["color"] == charts.PALETTE[3]
    assert spec["encoding"]["x"]["title"] == "Rebalance"


def test_exposure_chart_series_is_a_horizontal_bar_of_the_latest_weights() -> None:
    weights = pd.Series({"Alpha": 0.6, "Beta": 0.4})
    spec = charts.exposure_chart(weights).to_dict()
    assert spec["mark"]["type"] == "bar"
    assert spec["height"] == 260
    assert spec["encoding"]["y"]["sort"] == "-x"
    rows = spec["datasets"][spec["data"]["name"]]
    assert [(row["Manager"], row["Weight"], row["Date"]) for row in rows] == [
        ("Alpha", 0.6, "Latest"),
        ("Beta", 0.4, "Latest"),
    ]


def test_exposure_chart_frame_is_a_stacked_area_melted_by_manager() -> None:
    index = pd.to_datetime(["2024-01-31", "2024-02-29"])
    weights = pd.DataFrame({"Alpha": [0.6, 0.5], "Beta": [0.4, 0.5]}, index=index)
    spec = charts.exposure_chart(weights).to_dict()
    assert spec["mark"]["type"] == "area"
    rows = spec["datasets"][spec["data"]["name"]]
    assert sorted((row["Manager"], row["Weight"]) for row in rows) == [
        ("Alpha", 0.5),
        ("Alpha", 0.6),
        ("Beta", 0.4),
        ("Beta", 0.5),
    ]


def test_exposure_chart_none_and_empty_frame_give_an_empty_line_chart() -> None:
    for empty in (None, pd.DataFrame()):
        spec = charts.exposure_chart(empty).to_dict()
        assert spec["mark"]["type"] == "line"
        assert spec["datasets"][spec["data"]["name"]] == []
