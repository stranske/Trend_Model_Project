"""Monte Carlo plotting helpers for the Streamlit app."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .charts import PALETTE

_SHARPE_ALIASES = ("sharpe", "sharp", "sharperatio")
_MAX_DD_ALIASES = ("maxdrawdown", "maxdd", "drawdown", "max_drawdown")
_TERMINAL_ALIASES = (
    "terminalwealth",
    "terminalvalue",
    "finalwealth",
    "endingwealth",
    "terminal_wealth",
    "nav",
)


def _canonical_metric(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _resolve_metric(metrics: Iterable[str], aliases: Sequence[str]) -> str | None:
    alias_set = {_canonical_metric(alias) for alias in aliases}
    for metric in metrics:
        if _canonical_metric(metric) in alias_set:
            return metric
    return None


def _extract_results_frame(results: Any) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        return results
    if hasattr(results, "results_frame"):
        frame = getattr(results, "results_frame")
        if isinstance(frame, pd.DataFrame):
            return frame
    if isinstance(results, Mapping):
        for key in ("results_frame", "results", "frame"):
            frame = results.get(key)
            if isinstance(frame, pd.DataFrame):
                return frame
    return pd.DataFrame()


def _empty_chart() -> alt.Chart:
    return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()


def _metric_frame(results: Any, metric: str | None) -> pd.DataFrame:
    results_frame = _extract_results_frame(results)
    if results_frame.empty:
        return pd.DataFrame(columns=["Strategy", "Value"])
    if "strategy" not in results_frame.columns:
        return pd.DataFrame(columns=["Strategy", "Value"])

    metric_name = metric
    if metric_name is None:
        metric_names = [str(col) for col in results_frame.columns if col != "strategy"]
        metric_name = _resolve_metric(metric_names, _SHARPE_ALIASES)
        if metric_name is None:
            metric_name = _resolve_metric(metric_names, _MAX_DD_ALIASES)
        if metric_name is None:
            metric_name = _resolve_metric(metric_names, _TERMINAL_ALIASES)
        if metric_name is None:
            numeric_cols = [
                col
                for col in results_frame.columns
                if col != "strategy" and pd.api.types.is_numeric_dtype(results_frame[col])
            ]
            metric_name = numeric_cols[0] if numeric_cols else None

    if metric_name is None or metric_name not in results_frame.columns:
        return pd.DataFrame(columns=["Strategy", "Value"])

    frame = results_frame[["strategy", metric_name]].copy()
    frame = frame.rename(columns={"strategy": "Strategy", metric_name: "Value"})
    frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    return frame.dropna(subset=["Value"])


def sharpe_histogram(
    results: Any,
    *,
    metric: str | None = None,
    max_bins: int = 30,
    max_strategies: int = 8,
) -> alt.Chart:
    """Return a histogram chart of Sharpe distributions per strategy."""

    frame = _metric_frame(results, metric)
    if frame.empty:
        return _empty_chart()

    top_strategies = frame["Strategy"].value_counts().index[:max_strategies]
    frame = frame[frame["Strategy"].isin(top_strategies)]
    return (
        alt.Chart(frame)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("Value:Q", bin=alt.Bin(maxbins=max_bins), title="Sharpe"),
            y=alt.Y("count():Q", title="Count"),
            color=alt.Color("Strategy:N", scale=alt.Scale(range=PALETTE)),
            tooltip=["Strategy:N", "count():Q"],
        )
        .properties(height=260)
    )


def render_sharpe_histogram(
    results: Any,
    *,
    metric: str | None = None,
    max_bins: int = 30,
    max_strategies: int = 8,
) -> alt.Chart:
    chart = sharpe_histogram(
        results,
        metric=metric,
        max_bins=max_bins,
        max_strategies=max_strategies,
    )
    st.altair_chart(chart, use_container_width=True)
    return chart


def _nav_long_frame(nav_paths: pd.DataFrame, *, max_paths: int = 200) -> pd.DataFrame:
    if nav_paths.empty:
        return pd.DataFrame(columns=["Date", "Path", "NAV"])

    frame = nav_paths.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        names = [name or "" for name in frame.columns.names]
        if "path" in names:
            if "asset" in names:
                asset_level = names.index("asset")
                assets = frame.columns.get_level_values(asset_level)
                preferred = None
                for candidate in ("NAV", "nav", "wealth"):
                    if candidate in assets:
                        preferred = candidate
                        break
                if preferred is not None:
                    frame = frame.xs(preferred, level=asset_level, axis=1)
                else:
                    unique_assets = list(pd.unique(assets))
                    frame = frame.xs(unique_assets[0], level=asset_level, axis=1)
            path_level = names.index("path")
            frame.columns = frame.columns.get_level_values(path_level)
        else:
            frame.columns = ["_".join(map(str, col)) for col in frame.columns]

    if frame.shape[1] > max_paths:
        frame = frame.iloc[:, :max_paths]

    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[frame.index.notna()]
    index_name = frame.index.name or "index"
    melted = frame.reset_index().melt(id_vars=index_name, var_name="Path", value_name="NAV")
    melted = melted.rename(columns={index_name: "Date"})
    melted["NAV"] = pd.to_numeric(melted["NAV"], errors="coerce")
    return melted.dropna(subset=["Date", "NAV"])


def fan_chart(nav_paths: pd.DataFrame, *, max_paths: int = 200) -> alt.Chart:
    """Return a fan chart of NAV paths over time."""

    if not isinstance(nav_paths, pd.DataFrame):
        return _empty_chart()
    frame = _nav_long_frame(nav_paths, max_paths=max_paths)
    if frame.empty:
        return _empty_chart()

    return (
        alt.Chart(frame)
        .mark_line(opacity=0.2)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("NAV:Q", title="NAV", axis=alt.Axis(format=",.2f")),
            color=alt.Color("Path:N", legend=None),
            tooltip=["Path:N", alt.Tooltip("NAV:Q", format=",.2f")],
        )
        .properties(height=260)
    )


def render_fan_chart(nav_paths: pd.DataFrame, *, max_paths: int = 200) -> alt.Chart:
    chart = fan_chart(nav_paths, max_paths=max_paths)
    st.altair_chart(chart, use_container_width=True)
    return chart


def box_plot(
    results: Any,
    *,
    metric: str | None = None,
    max_strategies: int = 12,
) -> alt.Chart:
    """Return a box plot for strategy comparison."""

    frame = _metric_frame(results, metric)
    if frame.empty:
        return _empty_chart()

    top_strategies = frame["Strategy"].value_counts().index[:max_strategies]
    frame = frame[frame["Strategy"].isin(top_strategies)]

    return (
        alt.Chart(frame)
        .mark_boxplot(size=20)
        .encode(
            x=alt.X("Strategy:N", sort="-y", title="Strategy"),
            y=alt.Y("Value:Q", title="Metric"),
            color=alt.Color("Strategy:N", scale=alt.Scale(range=PALETTE), legend=None),
            tooltip=["Strategy:N", alt.Tooltip("Value:Q", format=",.2f")],
        )
        .properties(height=280)
    )


def render_box_plot(
    results: Any,
    *,
    metric: str | None = None,
    max_strategies: int = 12,
) -> alt.Chart:
    chart = box_plot(results, metric=metric, max_strategies=max_strategies)
    st.altair_chart(chart, use_container_width=True)
    return chart


def cdf_plot(
    results: Any,
    *,
    metric: str | None = None,
    max_strategies: int = 8,
) -> alt.Chart:
    """Return a cumulative distribution plot for outcomes."""

    frame = _metric_frame(results, metric)
    if frame.empty:
        return _empty_chart()

    top_strategies = frame["Strategy"].value_counts().index[:max_strategies]
    frame = frame[frame["Strategy"].isin(top_strategies)]
    frame = frame.sort_values(["Strategy", "Value"])
    frame["Rank"] = frame.groupby("Strategy").cumcount() + 1
    frame["Count"] = frame.groupby("Strategy")["Value"].transform("size")
    frame["CDF"] = frame["Rank"] / frame["Count"]

    return (
        alt.Chart(frame)
        .mark_line()
        .encode(
            x=alt.X("Value:Q", title="Outcome"),
            y=alt.Y("CDF:Q", title="CDF", axis=alt.Axis(format=".0%")),
            color=alt.Color("Strategy:N", scale=alt.Scale(range=PALETTE)),
            tooltip=["Strategy:N", alt.Tooltip("Value:Q", format=",.2f"), alt.Tooltip("CDF:Q", format=".0%")],
        )
        .properties(height=260)
    )


def render_cdf_plot(
    results: Any,
    *,
    metric: str | None = None,
    max_strategies: int = 8,
) -> alt.Chart:
    chart = cdf_plot(results, metric=metric, max_strategies=max_strategies)
    st.altair_chart(chart, use_container_width=True)
    return chart
