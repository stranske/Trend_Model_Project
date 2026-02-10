"""Plotly-based Monte Carlo plotting helpers for the Streamlit app."""

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
_RETURN_ALIASES = (
    "return",
    "annualreturn",
    "expectedreturn",
    "meanreturn",
    "cagr",
    "terminalwealth",
    "terminalvalue",
    "finalwealth",
    "endingwealth",
    "terminal_wealth",
    "nav",
)
_RISK_ALIASES = (
    "volatility",
    "vol",
    "stdev",
    "stddev",
    "sigma",
    "risk",
    "maxdrawdown",
    "maxdd",
    "drawdown",
    "max_drawdown",
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

    metric_name = _select_metric_name(results_frame, metric)
    if metric_name is None or metric_name not in results_frame.columns:
        return pd.DataFrame(columns=["Strategy", "Value"])

    frame = results_frame[["strategy", metric_name]].copy()
    frame = frame.rename(columns={"strategy": "Strategy", metric_name: "Value"})
    frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    return frame.dropna(subset=["Value"])


def _select_metric_name(results_frame: pd.DataFrame, metric: str | None) -> str | None:
    if metric:
        return metric if metric in results_frame.columns else None

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
    return metric_name if metric_name in results_frame.columns else None


def _select_metric_by_aliases(
    results_frame: pd.DataFrame,
    aliases: Sequence[str],
    *,
    fallback_numeric: bool = True,
) -> str | None:
    metric_names = [str(col) for col in results_frame.columns if col != "strategy"]
    metric_name = _resolve_metric(metric_names, aliases)
    if metric_name is None and fallback_numeric:
        numeric_cols = [
            col
            for col in results_frame.columns
            if col != "strategy" and pd.api.types.is_numeric_dtype(results_frame[col])
        ]
        metric_name = numeric_cols[0] if numeric_cols else None
    return metric_name if metric_name in results_frame.columns else None


def _metric_warning_message(results: Any, metric: str | None, label: str) -> str | None:
    results_frame = _extract_results_frame(results)
    if results_frame.empty:
        return f"{label} unavailable: results frame is missing."
    if "strategy" not in results_frame.columns:
        return f"{label} unavailable: required column 'strategy' is missing."

    metric_name = _select_metric_name(results_frame, metric)
    if metric_name is None:
        if metric:
            return f"{label} unavailable: required column '{metric}' is missing."
        return f"{label} unavailable: no numeric metric columns were found."

    metric_values = pd.to_numeric(results_frame[metric_name], errors="coerce")
    if metric_values.dropna().empty:
        return f"{label} unavailable: column '{metric_name}' has no numeric values."
    return None


def _metric_warning_message_for_aliases(
    results: Any,
    aliases: Sequence[str],
    label: str,
    *,
    fallback_numeric: bool = True,
) -> str | None:
    results_frame = _extract_results_frame(results)
    if results_frame.empty:
        return f"{label} unavailable: results frame is missing."
    if "strategy" not in results_frame.columns:
        return f"{label} unavailable: required column 'strategy' is missing."

    metric_name = _select_metric_by_aliases(
        results_frame,
        aliases,
        fallback_numeric=fallback_numeric,
    )
    if metric_name is None:
        return f"{label} unavailable: required metric columns were not found."

    metric_values = pd.to_numeric(results_frame[metric_name], errors="coerce")
    if metric_values.dropna().empty:
        return f"{label} unavailable: column '{metric_name}' has no numeric values."
    return None


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
    fig.update_traces(
        opacity=0.6, hovertemplate="Strategy=%{legendgroup}<br>Count=%{y}<extra></extra>"
    )
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
        warning = _metric_warning_message(results, metric, "Sharpe distribution chart")
        st.warning(warning or "Sharpe distribution chart unavailable: no data to display.")
    st.plotly_chart(chart, use_container_width=True)
    return chart


def path_distribution_chart(
    results: Any,
    *,
    metric: str | None = None,
    max_bins: int = 40,
    max_strategies: int = 8,
) -> go.Figure:
    """Return a histogram chart of terminal path outcomes."""

    results_frame = _extract_results_frame(results)
    if results_frame.empty or "strategy" not in results_frame.columns:
        return _empty_chart()

    metric_name = metric if metric and metric in results_frame.columns else None
    if metric_name is None:
        metric_name = _select_metric_by_aliases(results_frame, _TERMINAL_ALIASES)
    if metric_name is None:
        return _empty_chart()

    frame = results_frame[["strategy", metric_name]].copy()
    frame = frame.rename(columns={"strategy": "Strategy", metric_name: "Value"})
    frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    frame = frame.dropna(subset=["Value"])
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
    fig.update_traces(
        opacity=0.6, hovertemplate="Strategy=%{legendgroup}<br>Count=%{y}<extra></extra>"
    )
    fig.update_layout(
        height=260,
        xaxis_title=metric_name.replace("_", " ").title(),
        yaxis_title="Count",
        legend_title_text="Strategy",
        bargap=0.05,
    )
    return fig


def render_path_distribution_chart(
    results: Any,
    *,
    metric: str | None = None,
    max_bins: int = 40,
    max_strategies: int = 8,
) -> go.Figure:
    chart = path_distribution_chart(
        results,
        metric=metric,
        max_bins=max_bins,
        max_strategies=max_strategies,
    )
    if not chart.data:
        warning = _metric_warning_message_for_aliases(
            results,
            _TERMINAL_ALIASES,
            "Path distribution chart",
        )
        st.warning(warning or "Path distribution chart unavailable: no data to display.")
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
        warning = _metric_warning_message(results, metric, "Metric comparison chart")
        st.warning(warning or "Metric comparison chart unavailable: no data to display.")
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
        warning = _metric_warning_message(results, metric, "Outcome CDF chart")
        st.warning(warning or "Outcome CDF chart unavailable: no data to display.")
    st.plotly_chart(chart, use_container_width=True)
    return chart


def risk_return_chart(
    results: Any,
    *,
    return_metric: str | None = None,
    risk_metric: str | None = None,
    max_strategies: int = 12,
) -> go.Figure:
    """Return a risk-return scatter plot across strategies."""

    results_frame = _extract_results_frame(results)
    if results_frame.empty or "strategy" not in results_frame.columns:
        return _empty_chart()

    numeric_cols = [
        col
        for col in results_frame.columns
        if col != "strategy" and pd.api.types.is_numeric_dtype(results_frame[col])
    ]
    return_metric_name = (
        return_metric if return_metric and return_metric in results_frame.columns else None
    )
    if return_metric_name is None:
        return_metric_name = _select_metric_by_aliases(
            results_frame,
            _RETURN_ALIASES,
            fallback_numeric=False,
        )
    if return_metric_name is None and numeric_cols:
        return_metric_name = numeric_cols[0]

    risk_metric_name = risk_metric if risk_metric and risk_metric in results_frame.columns else None
    if risk_metric_name is None:
        risk_metric_name = _select_metric_by_aliases(
            results_frame,
            _RISK_ALIASES,
            fallback_numeric=False,
        )
    if risk_metric_name is None:
        risk_metric_name = next(
            (col for col in numeric_cols if col != return_metric_name),
            None,
        )

    if return_metric_name is None or risk_metric_name is None:
        return _empty_chart()

    frame = results_frame[["strategy", return_metric_name, risk_metric_name]].copy()
    frame = frame.rename(
        columns={
            "strategy": "Strategy",
            return_metric_name: "Return",
            risk_metric_name: "Risk",
        }
    )
    frame["Return"] = pd.to_numeric(frame["Return"], errors="coerce")
    frame["Risk"] = pd.to_numeric(frame["Risk"], errors="coerce")
    if _canonical_metric(risk_metric_name) in {
        _canonical_metric(alias) for alias in _MAX_DD_ALIASES
    }:
        frame["Risk"] = frame["Risk"].abs()
    frame = frame.dropna(subset=["Return", "Risk"])
    if frame.empty:
        return _empty_chart()

    top_strategies = frame["Strategy"].value_counts().index[:max_strategies]
    frame = frame[frame["Strategy"].isin(top_strategies)]

    grouped = frame.groupby("Strategy", as_index=False)[["Return", "Risk"]].mean()
    fig = px.scatter(
        grouped,
        x="Risk",
        y="Return",
        color="Strategy",
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(
        marker=dict(size=10, opacity=0.8),
        hovertemplate="Strategy=%{legendgroup}<br>Risk=%{x:,.3f}<br>Return=%{y:,.3f}<extra></extra>",
    )
    fig.update_layout(
        height=260,
        xaxis_title=risk_metric_name.replace("_", " ").title(),
        yaxis_title=return_metric_name.replace("_", " ").title(),
        legend_title_text="Strategy",
    )
    return fig


def render_risk_return_chart(
    results: Any,
    *,
    return_metric: str | None = None,
    risk_metric: str | None = None,
    max_strategies: int = 12,
) -> go.Figure:
    chart = risk_return_chart(
        results,
        return_metric=return_metric,
        risk_metric=risk_metric,
        max_strategies=max_strategies,
    )
    if not chart.data:
        warning = _metric_warning_message_for_aliases(
            results,
            _RETURN_ALIASES,
            "Risk-return chart",
        )
        if warning is None:
            warning = _metric_warning_message_for_aliases(
                results,
                _RISK_ALIASES,
                "Risk-return chart",
            )
        st.warning(warning or "Risk-return chart unavailable: no data to display.")
    st.plotly_chart(chart, use_container_width=True)
    return chart
