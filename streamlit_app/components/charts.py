"""Shared chart helpers to keep Streamlit visuals stylistically aligned."""

from __future__ import annotations

from typing import Sequence

import altair as alt
import pandas as pd

PALETTE = [
    "#005f73",
    "#0a9396",
    "#94d2bd",
    "#ee9b00",
    "#ca6702",
    "#bb3e03",
]


def _base_chart() -> alt.Chart:
    return (
        alt.Chart()
        .configure_axis(grid=False, labelColor="#1f2933", titleColor="#1f2933")
        .configure_legend(title=None, orient="top")
        .configure_view(strokeOpacity=0)
    )


def line_chart(
    data: pd.DataFrame,
    *,
    x: str,
    value_fields: Sequence[str],
    title: str,
    y_title: str,
    y_format: str = ".2f",
) -> alt.Chart:
    source = data.reset_index(drop=False)
    melted = source.melt(id_vars=[x], value_vars=list(value_fields), var_name="Series", value_name="Value")
    chart = (
        alt.Chart(melted)
        .mark_line(point=False)
        .encode(
            x=alt.X(f"{x}:T", title="Date"),
            y=alt.Y("Value:Q", title=y_title, axis=alt.Axis(format=y_format)),
            color=alt.Color("Series:N", scale=alt.Scale(range=PALETTE)),
        )
        .properties(title=title)
    )
    return _base_chart() + chart


def area_chart(
    data: pd.DataFrame,
    *,
    x: str,
    value_fields: Sequence[str],
    title: str,
    y_title: str,
) -> alt.Chart:
    source = data.reset_index(drop=False)
    melted = source.melt(id_vars=[x], value_vars=list(value_fields), var_name="Series", value_name="Value")
    chart = (
        alt.Chart(melted)
        .mark_area(opacity=0.7)
        .encode(
            x=alt.X(f"{x}:T", title="Date"),
            y=alt.Y("Value:Q", stack="normalize", title=y_title),
            color=alt.Color("Series:N", scale=alt.Scale(range=PALETTE)),
        )
        .properties(title=title)
    )
    return _base_chart() + chart


def bar_chart(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    y_title: str,
    y_format: str = ".2f",
) -> alt.Chart:
    source = data.reset_index(drop=False)
    chart = (
        alt.Chart(source)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:T", title="Date"),
            y=alt.Y(f"{y}:Q", title=y_title, axis=alt.Axis(format=y_format)),
            color=alt.value(PALETTE[0]),
        )
        .properties(title=title)
    )
    return _base_chart() + chart


def category_bar_chart(
    data: pd.DataFrame,
    *,
    category: str,
    value: str,
    title: str,
    value_format: str = ".2f",
) -> alt.Chart:
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(f"{value}:Q", title=value, axis=alt.Axis(format=value_format)),
            y=alt.Y(f"{category}:N", sort="-x", title=None),
            color=alt.value(PALETTE[1]),
        )
        .properties(title=title)
    )
    return _base_chart() + chart


__all__ = [
    "line_chart",
    "area_chart",
    "bar_chart",
    "category_bar_chart",
    "PALETTE",
]
