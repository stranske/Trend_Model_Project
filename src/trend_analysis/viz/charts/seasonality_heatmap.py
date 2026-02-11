"""Seasonality heatmap for canonical Monte Carlo path data."""

from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    import streamlit as st
except Exception:  # pragma: no cover - streamlit is optional outside app runtime
    st = None


def _cache_data(
    *args: object, **kwargs: object
) -> Callable[[Callable[..., object]], Callable[..., object]]:
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
def _prepare_seasonality_matrix(paths: pd.DataFrame) -> pd.DataFrame:
    wide_nav = _to_nav_wide(paths).ffill()
    if wide_nav.empty:
        return pd.DataFrame()

    returns = wide_nav.pct_change().dropna(how="all")
    if returns.empty:
        return pd.DataFrame()
    monthly = returns.mean(axis=1)
    frame = pd.DataFrame({"value": monthly})
    frame["year"] = frame.index.year
    frame["month"] = frame.index.month
    return frame.pivot_table(index="year", columns="month", values="value", aggfunc="mean")


def build_figure(paths: pd.DataFrame) -> go.Figure:
    """Build a monthly seasonality heatmap from canonical ``make_paths`` output."""

    seasonality = _prepare_seasonality_matrix(paths)
    if seasonality.empty:
        return go.Figure()
    fig = px.imshow(
        seasonality.sort_index(),
        color_continuous_scale="RdYlGn",
        aspect="auto",
        labels={"x": "Month", "y": "Year", "color": "Mean return"},
    )
    fig.update_layout(height=320, title="Monthly Seasonality Heatmap")
    return fig


__all__ = ["build_figure"]
