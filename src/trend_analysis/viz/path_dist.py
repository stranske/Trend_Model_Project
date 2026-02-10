"""Path distribution visualizations.

Focused scope:
- Derive a terminal-value distribution from simulated NAV paths.
- Provide a Plotly helper that renders the histogram with optional quantile markers.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
import plotly.graph_objects as go

from .theme import apply_theme
from .utils import DEFAULT_COLORS, coerce_frame, ensure_non_empty, validate_quantiles

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


def _select_terminal_values(
    nav_paths: pd.DataFrame | dict[str, Sequence[float]],
    *,
    max_paths: int | None,
) -> pd.Series:
    """Normalize NAV paths into a Series of terminal values."""

    frame = coerce_frame(nav_paths, name="nav_paths")

    if isinstance(frame.columns, pd.MultiIndex):
        names = [name or "" for name in frame.columns.names]
        unexpected = [name for name in names if name not in {"", "asset", "path"}]
        if unexpected:
            raise ValueError(
                "nav_paths MultiIndex levels must be named 'asset' and/or 'path'"
            )
        if "asset" in names:
            asset_level = names.index("asset")
            assets = frame.columns.get_level_values(asset_level)
            preferred = None
            for candidate in ("NAV", "nav", "wealth"):
                if candidate in assets:
                    preferred = candidate
                    break
            if preferred is None and len(assets) > 0:
                preferred = assets[0]
            if preferred is not None:
                frame = frame.xs(preferred, level=asset_level, axis=1)

        if "path" in names:
            path_level = names.index("path")
            frame.columns = frame.columns.get_level_values(path_level)
        else:
            frame.columns = ["_".join(map(str, col)) for col in frame.columns]

    if max_paths is not None and frame.shape[1] > max_paths:
        frame = frame.iloc[:, :max_paths]

    frame = frame.sort_index()
    if not isinstance(frame.index, pd.DatetimeIndex):
        converted = pd.to_datetime(frame.index, errors="coerce")
        if converted.notna().any():
            frame = frame.copy()
            frame.index = converted
            frame = frame[frame.index.notna()]

    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    ensure_non_empty("nav_paths", frame)

    terminal = frame.ffill().iloc[-1].dropna()
    ensure_non_empty("terminal_values", terminal)
    terminal.name = "terminal_value"
    return terminal


def terminal_distribution(
    nav_paths: pd.DataFrame | dict[str, Sequence[float]],
    *,
    max_paths: int | None = 1000,
) -> pd.Series:
    """Return a Series of terminal values across simulated paths."""

    return _select_terminal_values(nav_paths, max_paths=max_paths)


def make(
    nav_paths: pd.DataFrame | dict[str, Sequence[float]],
    *,
    bins: int = 40,
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
    max_paths: int | None = 1000,
    title: str | None = "Path Distribution",
) -> go.Figure:
    """Create a histogram of terminal path values."""

    terminal = _select_terminal_values(nav_paths, max_paths=max_paths)
    q_values = validate_quantiles(quantiles)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=terminal,
            nbinsx=bins,
            marker=dict(color=DEFAULT_COLORS[0]),
            name="Terminal values",
        )
    )

    if q_values:
        for idx, q in enumerate(sorted(set(q_values))):
            q_value = float(terminal.quantile(q))
            fig.add_shape(
                type="line",
                x0=q_value,
                x1=q_value,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(
                    color=DEFAULT_COLORS[(idx + 1) % len(DEFAULT_COLORS)], dash="dash"
                ),
            )
            fig.add_annotation(
                x=q_value,
                y=1.02,
                yref="paper",
                text=f"{int(round(q * 100))}%",
                showarrow=False,
                font=dict(color=DEFAULT_COLORS[(idx + 1) % len(DEFAULT_COLORS)]),
            )

    fig.update_layout(
        title=title,
        xaxis_title="Terminal Value",
        yaxis_title="Count",
        bargap=0.05,
    )

    return apply_theme(fig)


__all__ = ["DEFAULT_QUANTILES", "terminal_distribution", "make"]
