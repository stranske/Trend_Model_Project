"""Common chart helpers for the Streamlit app."""

from __future__ import annotations

import altair as alt
import pandas as pd

PALETTE = [
    "#2563EB",  # blue
    "#F97316",  # orange
    "#10B981",  # green
    "#6366F1",  # indigo
    "#EC4899",  # pink
]


def _series_to_frame(series: pd.Series, label: str) -> pd.DataFrame:
    if series is None or series.empty:
        return pd.DataFrame(columns=["Date", label])
    frame = series.reset_index()
    frame.columns = ["Date", label]
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    return frame.dropna(subset=["Date"])


def equity_chart(equity: pd.Series) -> alt.Chart:
    frame = _series_to_frame(equity, "Equity")
    return (
        alt.Chart(frame)
        .mark_line(color=PALETTE[0], strokeWidth=2)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Equity:Q", title="Equity", axis=alt.Axis(format=",.2f")),
            tooltip=["Date:T", alt.Tooltip("Equity:Q", format=",.2f")],
        )
        .properties(height=280)
    )


def drawdown_chart(drawdown: pd.Series) -> alt.Chart:
    frame = _series_to_frame(drawdown, "Drawdown")
    return (
        alt.Chart(frame)
        .mark_area(color=PALETTE[1], opacity=0.45)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Drawdown:Q", title="Drawdown", axis=alt.Axis(format=".0%")),
            tooltip=["Date:T", alt.Tooltip("Drawdown:Q", format=".1%")],
        )
        .properties(height=220)
    )


def rolling_sharpe_chart(rolling_sharpe: pd.Series) -> alt.Chart:
    frame = _series_to_frame(rolling_sharpe, "Sharpe")
    return (
        alt.Chart(frame)
        .mark_line(color=PALETTE[2], strokeDash=[6, 4], strokeWidth=2)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Sharpe:Q", title="Rolling Sharpe", axis=alt.Axis(format=",.2f")),
            tooltip=["Date:T", alt.Tooltip("Sharpe:Q", format=",.2f")],
        )
        .properties(height=220)
    )


def turnover_chart(turnover: pd.Series) -> alt.Chart:
    frame = _series_to_frame(turnover, "Turnover")
    return (
        alt.Chart(frame)
        .mark_bar(color=PALETTE[3])
        .encode(
            x=alt.X("Date:T", title="Rebalance"),
            y=alt.Y("Turnover:Q", title="Turnover", axis=alt.Axis(format=".0%")),
            tooltip=["Date:T", alt.Tooltip("Turnover:Q", format=".1%")],
        )
        .properties(height=220)
    )


def exposure_chart(weights: pd.DataFrame | pd.Series) -> alt.Chart:
    if weights is None:
        return alt.Chart(pd.DataFrame(columns=["Date", "Manager", "Weight"])).mark_line()

    if isinstance(weights, pd.Series):
        frame = pd.DataFrame({"Manager": weights.index, "Weight": weights.values})
        frame["Date"] = "Latest"
        return (
            alt.Chart(frame)
            .mark_bar()
            .encode(
                x=alt.X("Weight:Q", title="Weight", axis=alt.Axis(format=".0%")),
                y=alt.Y("Manager:N", sort="-x", title="Manager"),
                tooltip=["Manager:N", alt.Tooltip("Weight:Q", format=".1%")],
                color=alt.Color("Manager:N", scale=alt.Scale(range=PALETTE)),
            )
            .properties(height=260)
        )

    if weights.empty:
        return alt.Chart(pd.DataFrame(columns=["Date", "Manager", "Weight"])).mark_line()

    frame = weights.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.reset_index().melt(id_vars="index", var_name="Manager", value_name="Weight")
    frame = frame.rename(columns={"index": "Date"}).dropna(subset=["Date"])

    return (
        alt.Chart(frame)
        .mark_area()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Weight:Q", stack="normalize", axis=alt.Axis(format=".0%")),
            color=alt.Color("Manager:N", scale=alt.Scale(range=PALETTE)),
            tooltip=["Date:T", "Manager:N", alt.Tooltip("Weight:Q", format=".1%")],
        )
        .properties(height=260)
    )
