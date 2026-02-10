"""Monte Carlo plotting helpers for the Streamlit app."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


def _empty_chart() -> go.Figure:
    return go.Figure()


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
) -> go.Figure:
    """Return a histogram chart of Sharpe distributions per strategy."""

    frame = _metric_frame(results, metric)
    if frame.empty:
        return _empty_chart()

    top_strategies = frame["Strategy"].value_counts().index[:max_strategies]
    frame = frame[frame["Strategy"].isin(top_strategies)]
    fig = px.histogram(
        frame,
        x="Value",
        color="Strategy",
        nbins=max_bins,
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(opacity=0.6, hovertemplate="Strategy=%{legendgroup}<br>Count=%{y}<extra></extra>")
    fig.update_layout(
        height=260,
        xaxis_title="Sharpe",
        yaxis_title="Count",
        legend_title_text="Strategy",
        bargap=0.05,
    )
    return fig


def render_sharpe_histogram(
    results: Any,
    *,
    metric: str | None = None,
    max_bins: int = 30,
    max_strategies: int = 8,
) -> go.Figure:
    chart = sharpe_histogram(
        results,
        metric=metric,
        max_bins=max_bins,
        max_strategies=max_strategies,
    )
    if not chart.data:
        st.warning("Sharpe distribution chart unavailable: missing strategy metrics.")
    st.plotly_chart(chart, use_container_width=True)
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


def _nav_wide_frame(nav_paths: pd.DataFrame, *, max_paths: int = 200) -> pd.DataFrame:
    if nav_paths.empty:
        return pd.DataFrame()

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
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(axis=0, how="all").dropna(axis=1, how="all")


def fan_chart(nav_paths: pd.DataFrame, *, max_paths: int = 200) -> go.Figure:
    """Return a fan chart of NAV paths over time using quantile bands."""

    if not isinstance(nav_paths, pd.DataFrame):
        return _empty_chart()
    frame = _nav_wide_frame(nav_paths, max_paths=max_paths)
    if frame.empty:
        return _empty_chart()

    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    q_frame = frame.quantile(quantiles, axis=1).T
    q_frame.columns = [f"q{int(q * 100)}" for q in quantiles]
    q_frame = q_frame.sort_index()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=q_frame.index,
            y=q_frame["q95"],
            line=dict(color="rgba(31, 119, 180, 0.2)"),
            name="95th percentile",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=q_frame.index,
            y=q_frame["q5"],
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.15)",
            line=dict(color="rgba(31, 119, 180, 0.2)"),
            name="5-95%",
            hoverinfo="skip",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=q_frame.index,
            y=q_frame["q75"],
            line=dict(color="rgba(44, 160, 44, 0.3)"),
            name="75th percentile",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=q_frame.index,
            y=q_frame["q25"],
            fill="tonexty",
            fillcolor="rgba(44, 160, 44, 0.18)",
            line=dict(color="rgba(44, 160, 44, 0.3)"),
            name="25-75%",
            hoverinfo="skip",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=q_frame.index,
            y=q_frame["q50"],
            line=dict(color=PALETTE[0] if PALETTE else "#1f77b4", width=2),
            name="Median",
        )
    )

    fig.update_layout(
        height=260,
        xaxis_title="Date",
        yaxis_title="NAV",
        yaxis_tickformat=",.2f",
        legend_title_text="Bands",
    )
    return fig


def render_fan_chart(nav_paths: pd.DataFrame, *, max_paths: int = 200) -> go.Figure:
    chart = fan_chart(nav_paths, max_paths=max_paths)
    if not chart.data:
        st.warning("Fan chart unavailable: NAV path data is missing.")
    st.plotly_chart(chart, use_container_width=True)
    return chart


def box_plot(
    results: Any,
    *,
    metric: str | None = None,
    max_strategies: int = 12,
) -> go.Figure:
    """Return a box plot for strategy comparison."""

    frame = _metric_frame(results, metric)
    if frame.empty:
        return _empty_chart()

    top_strategies = frame["Strategy"].value_counts().index[:max_strategies]
    frame = frame[frame["Strategy"].isin(top_strategies)]

    fig = px.box(
        frame,
        x="Strategy",
        y="Value",
        color="Strategy",
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(
        height=280,
        xaxis_title="Strategy",
        yaxis_title="Metric",
        showlegend=False,
    )
    fig.update_traces(hovertemplate="Strategy=%{x}<br>Value=%{y:,.2f}<extra></extra>")
    return fig


def render_box_plot(
    results: Any,
    *,
    metric: str | None = None,
    max_strategies: int = 12,
) -> go.Figure:
    chart = box_plot(results, metric=metric, max_strategies=max_strategies)
    if not chart.data:
        st.warning("Metric comparison chart unavailable: missing strategy metrics.")
    st.plotly_chart(chart, use_container_width=True)
    return chart


def cdf_plot(
    results: Any,
    *,
    metric: str | None = None,
    max_strategies: int = 8,
) -> go.Figure:
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

    fig = px.line(
        frame,
        x="Value",
        y="CDF",
        color="Strategy",
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(
        height=260,
        xaxis_title="Outcome",
        yaxis_title="CDF",
        yaxis_tickformat=".0%",
        legend_title_text="Strategy",
    )
    fig.update_traces(
        hovertemplate="Strategy=%{legendgroup}<br>Value=%{x:,.2f}<br>CDF=%{y:.0%}<extra></extra>"
    )
    return fig


def render_cdf_plot(
    results: Any,
    *,
    metric: str | None = None,
    max_strategies: int = 8,
) -> go.Figure:
    chart = cdf_plot(results, metric=metric, max_strategies=max_strategies)
    if not chart.data:
        st.warning("Outcome CDF chart unavailable: missing strategy metrics.")
    st.plotly_chart(chart, use_container_width=True)
    return chart
