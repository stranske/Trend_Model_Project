"""Correlation heatmap chart for canonical Monte Carlo path data."""

from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from trend_analysis.viz.adapters import path_correlations

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


@_cache_data(show_spinner=False)
def _prepare_corr_matrix(paths: pd.DataFrame, *, window: int | None = 60) -> pd.DataFrame:
    return path_correlations(paths, window=window)


def build_figure(data: pd.DataFrame, *, window: int | None = 60) -> go.Figure:
    """Build a correlation heatmap from canonical ``make_paths`` output."""

    corr = _prepare_corr_matrix(data, window=window)
    if corr.empty:
        return go.Figure()
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu",
        zmin=-1.0,
        zmax=1.0,
        labels={"x": "Path", "y": "Path", "color": "Correlation"},
        aspect="auto",
    )
    fig.update_layout(height=380, title="Path Correlation Heatmap")
    return fig


__all__ = ["build_figure"]
