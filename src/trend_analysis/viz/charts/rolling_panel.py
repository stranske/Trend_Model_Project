"""Rolling diagnostics panel for canonical Monte Carlo path data."""

from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.graph_objects as go

from trend_analysis.viz.adapters import rolling_stats

try:
    import streamlit as st
except Exception:  # pragma: no cover - streamlit is optional outside app runtime
    st = None


def _cache_data(*args: object, **kwargs: object) -> Callable[[Callable[..., object]], Callable[..., object]]:
    cache_data = getattr(st, "cache_data", None) if st is not None else None
    if callable(cache_data):
        return cache_data(*args, **kwargs)

    def _identity(func: Callable[..., object]) -> Callable[..., object]:
        return func

    return _identity


def _to_nav_wide(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    nav = pd.to_numeric(paths["nav"], errors="coerce")
    wide = nav.unstack("path")
    wide.index = pd.to_datetime(wide.index, errors="coerce")
    wide = wide[wide.index.notna()]
    return wide.sort_index()


@_cache_data(show_spinner=False)
def _prepare_panel_series(
    paths: pd.DataFrame,
    *,
    window: int = 12,
    periods_per_year: int = 12,
    max_paths: int = 6,
) -> pd.DataFrame:
    rolling = rolling_stats(paths, window=window, periods_per_year=periods_per_year)
    if rolling.empty:
        return pd.DataFrame()

    wide_nav = _to_nav_wide(paths).ffill()
    if wide_nav.empty:
        return pd.DataFrame()

    drawdown = wide_nav / wide_nav.cummax() - 1.0
    roll_std = rolling["rolling_std"].unstack("path")
    roll_sharpe = rolling["rolling_sharpe"].unstack("path")
    selected_paths = [col for col in wide_nav.columns[:max_paths]]

    records: list[dict[str, object]] = []
    for path_id in selected_paths:
        if path_id in roll_sharpe.columns:
            for date, value in roll_sharpe[path_id].items():
                records.append(
                    {"date": date, "path": path_id, "series": "rolling_sharpe", "value": value}
                )
        if path_id in roll_std.columns:
            for date, value in roll_std[path_id].items():
                records.append({"date": date, "path": path_id, "series": "rolling_vol", "value": value})
        if path_id in drawdown.columns:
            for date, value in drawdown[path_id].items():
                records.append({"date": date, "path": path_id, "series": "drawdown", "value": value})

    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["date", "value"]).sort_values(["path", "series", "date"])


def build_figure(
    paths: pd.DataFrame,
    *,
    window: int = 12,
    periods_per_year: int = 12,
    max_paths: int = 6,
) -> go.Figure:
    """Build rolling diagnostics panel from canonical ``make_paths`` output."""

    panel = _prepare_panel_series(
        paths,
        window=window,
        periods_per_year=periods_per_year,
        max_paths=max_paths,
    )
    if panel.empty:
        return go.Figure()

    fig = go.Figure()
    for (path_id, series_name), subset in panel.groupby(["path", "series"], sort=True):
        path_label = f"Path {path_id}"
        series_label = {
            "rolling_sharpe": "rolling Sharpe",
            "rolling_vol": "rolling vol",
            "drawdown": "drawdown",
        }.get(str(series_name), str(series_name))
        fig.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["value"],
                mode="lines",
                name=f"{path_label} {series_label}",
                legendgroup=str(path_id),
            )
        )

    fig.update_layout(
        height=420,
        title="Rolling Diagnostics Panel",
        xaxis_title="Date",
        yaxis_title="Value",
    )
    return fig


__all__ = ["build_figure"]
