"""Sharpe ladder visualizations from adapter-style summary frames.

Expected input columns:
- ``strategy``: strategy label.
- ``sharpe`` (or configurable metric): numeric Sharpe ratio value.

Optional columns such as ``fold_id``/``fold_label``/``paths`` are preserved
only for preprocessing; chart rendering uses strategy + metric.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import plotly.graph_objects as go

from .theme import apply_theme
from .utils import coerce_frame, ensure_non_empty

REQUIRED_COLUMNS: tuple[str, ...] = ("strategy", "sharpe")
DEFAULT_POSITIVE_COLOR = "#2a9d8f"
DEFAULT_NEGATIVE_COLOR = "#e76f51"


def prepare_sharpe_ladder(
    summary: pd.DataFrame | Mapping[str, list[Any]],
    *,
    metric: str = "sharpe",
    aggregate_duplicates: bool = True,
    ascending: bool = True,
) -> pd.DataFrame:
    """Normalize adapter summary data into a ladder-ready frame.

    Parameters
    ----------
    summary:
        Adapter summary-like frame containing ``strategy`` and ``metric``.
    metric:
        Metric column to chart. Defaults to ``"sharpe"``.
    aggregate_duplicates:
        If ``True``, duplicate strategies are averaged to one row.
    ascending:
        Sort order for the resulting ladder values.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``strategy`` and ``metric`` sorted for plotting.
    """

    frame = coerce_frame(summary, name="summary").copy()
    ensure_non_empty("summary", frame)

    if "strategy" not in frame.columns:
        raise ValueError("summary must include a 'strategy' column")
    if metric not in frame.columns:
        raise ValueError(f"summary must include a '{metric}' column")

    frame["strategy"] = frame["strategy"].astype("string")
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=["strategy", metric])
    ensure_non_empty("summary", frame)

    ladder = frame[["strategy", metric]].copy()
    if aggregate_duplicates:
        ladder = ladder.groupby("strategy", as_index=False)[metric].mean()
    ladder = ladder.sort_values(metric, ascending=ascending).reset_index(drop=True)
    return ladder


def make(
    summary: pd.DataFrame | Mapping[str, list[Any]],
    *,
    metric: str = "sharpe",
    title: str | None = "Sharpe Ladder",
    xaxis_title: str | None = "Sharpe Ratio",
    yaxis_title: str | None = "Strategy",
    show_values: bool = True,
    positive_color: str = DEFAULT_POSITIVE_COLOR,
    negative_color: str = DEFAULT_NEGATIVE_COLOR,
    aggregate_duplicates: bool = True,
) -> go.Figure:
    """Create a Plotly Sharpe ladder from adapter-style summary output.

    Required input columns are ``strategy`` and the selected ``metric``.
    Styling is configurable via title/axis labels and bar colors.
    """

    ladder = prepare_sharpe_ladder(
        summary,
        metric=metric,
        aggregate_duplicates=aggregate_duplicates,
        ascending=True,
    )

    colors = [positive_color if float(value) >= 0.0 else negative_color for value in ladder[metric]]

    fig = go.Figure(
        data=[
            go.Bar(
                x=ladder[metric],
                y=ladder["strategy"],
                orientation="h",
                marker=dict(color=colors),
                text=[f"{float(v):.2f}" for v in ladder[metric]] if show_values else None,
                textposition="outside" if show_values else None,
                name=metric,
            )
        ]
    )
    fig.add_vline(x=0.0, line_dash="dash", line_color="#6b7280", opacity=0.8)
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=False,
    )
    fig.update_yaxes(type="category")
    return apply_theme(fig)


__all__ = [
    "REQUIRED_COLUMNS",
    "DEFAULT_POSITIVE_COLOR",
    "DEFAULT_NEGATIVE_COLOR",
    "prepare_sharpe_ladder",
    "make",
]
