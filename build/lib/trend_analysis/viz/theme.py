"""Plotly theming helpers for trend analysis visualizations."""

from __future__ import annotations

from typing import Any, Mapping

import plotly.graph_objects as go

from .utils import DEFAULT_COLORS

DEFAULT_FONT_FAMILY = "Open Sans"
DEFAULT_FONT_COLOR = "#1f2933"
DEFAULT_AXIS_COLOR = "#52616b"
DEFAULT_GRID_COLOR = "#e5e7eb"


def base_layout() -> dict[str, Any]:
    """Return a base layout dictionary for Plotly figures."""

    return dict(
        font=dict(family=DEFAULT_FONT_FAMILY, color=DEFAULT_FONT_COLOR, size=12),
        colorway=list(DEFAULT_COLORS),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0.0),
        margin=dict(l=60, r=40, t=60, b=50),
        hovermode="x unified",
        xaxis=dict(
            showgrid=True,
            gridcolor=DEFAULT_GRID_COLOR,
            linecolor=DEFAULT_AXIS_COLOR,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=DEFAULT_GRID_COLOR,
            linecolor=DEFAULT_AXIS_COLOR,
            zeroline=False,
        ),
    )


def trend_template() -> go.layout.Template:
    """Return a Plotly template with Trend Analysis defaults."""

    return go.layout.Template(layout=base_layout())


def apply_theme(
    fig: go.Figure,
    *,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    legend_title: str | None = None,
    extra_layout: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Apply the Trend Analysis theme to a Plotly figure."""

    fig.update_layout(template=trend_template())
    if title is not None:
        fig.update_layout(title=title)
    if xaxis_title is not None:
        fig.update_xaxes(title_text=xaxis_title)
    if yaxis_title is not None:
        fig.update_yaxes(title_text=yaxis_title)
    if legend_title is not None:
        fig.update_layout(legend_title_text=legend_title)
    if extra_layout:
        fig.update_layout(**dict(extra_layout))
    return fig


__all__ = [
    "DEFAULT_FONT_FAMILY",
    "DEFAULT_FONT_COLOR",
    "DEFAULT_AXIS_COLOR",
    "DEFAULT_GRID_COLOR",
    "base_layout",
    "trend_template",
    "apply_theme",
]
