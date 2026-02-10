"""Fan chart visualizations."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
import plotly.graph_objects as go

from .utils import DEFAULT_COLORS, coerce_frame, ensure_non_empty, hex_to_rgba, quantile_bands, quantiles_over_columns

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


def _select_nav_paths(
    nav_paths: pd.DataFrame | dict[str, Sequence[float]],
    *,
    max_paths: int | None,
) -> pd.DataFrame:
    """Normalize NAV paths into a numeric DataFrame with a clean index."""

    frame = coerce_frame(nav_paths, name="nav_paths")

    if isinstance(frame.columns, pd.MultiIndex):
        names = [name or "" for name in frame.columns.names]
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
    return frame


def _median_quantile(quantiles: Iterable[float]) -> float | None:
    """Pick the quantile closest to 0.5 (median)."""

    q_values = tuple(float(q) for q in quantiles)
    if not q_values:
        return None
    return min(q_values, key=lambda q: abs(q - 0.5))


def make(
    nav_paths: pd.DataFrame | dict[str, Sequence[float]],
    *,
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
    max_paths: int | None = 200,
    show_paths: bool = False,
    title: str | None = "Fan Chart",
) -> go.Figure:
    """Create a fan chart from simulated NAV paths."""

    frame = _select_nav_paths(nav_paths, max_paths=max_paths)
    quantiles_frame = quantiles_over_columns(frame, quantiles)
    bands = quantile_bands(quantiles)

    fig = go.Figure()
    x_vals = quantiles_frame.index

    base_color = DEFAULT_COLORS[0]
    if bands:
        alpha_step = 0.6 / max(len(bands), 1)
    else:
        alpha_step = 0.2

    for idx, band in enumerate(bands):
        upper = quantiles_frame[band.upper]
        lower = quantiles_frame[band.lower]
        fill_alpha = min(0.15 + alpha_step * idx, 0.7)
        fill_color = hex_to_rgba(base_color, fill_alpha)

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=fill_color,
                name=band.label(),
                hoverinfo="skip",
            )
        )

    median_q = _median_quantile(quantiles_frame.columns)
    if median_q is not None:
        median = quantiles_frame[median_q]
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=median,
                mode="lines",
                line=dict(color=base_color, width=2),
                name="Median",
            )
        )

    if show_paths:
        path_color = hex_to_rgba(DEFAULT_COLORS[-1], 0.25)
        for col in frame.columns:
            fig.add_trace(
                go.Scatter(
                    x=frame.index,
                    y=frame[col],
                    mode="lines",
                    line=dict(color=path_color, width=1),
                    name=str(col),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title="Quantile Band",
        template="plotly_white",
    )
    return fig


__all__ = ["make"]
