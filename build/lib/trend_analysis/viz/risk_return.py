"""Risk/return visualizations.

Focused scope:
- Summarize return and volatility per strategy.
- Provide a Plotly helper to render a risk/return scatter.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .theme import apply_theme
from .utils import coerce_frame, ensure_non_empty

DEFAULT_PERIODS_PER_YEAR = 12.0


def risk_return_summary(
    returns: pd.DataFrame | Mapping[str, Sequence[float]],
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Compute annualized return/volatility statistics."""

    frame = coerce_frame(returns, name="returns").apply(pd.to_numeric, errors="coerce")
    ensure_non_empty("returns", frame)

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    mean = frame.mean(skipna=True)
    vol = frame.std(skipna=True, ddof=0)
    ann_return = mean * periods_per_year
    ann_vol = vol * math.sqrt(periods_per_year)
    excess = ann_return - risk_free_rate
    sharpe = pd.Series(
        np.where(ann_vol > 0, excess / ann_vol, np.nan),
        index=frame.columns,
        name="sharpe",
    )

    summary = pd.DataFrame(
        {
            "return": ann_return,
            "volatility": ann_vol,
            "sharpe": sharpe,
        }
    )
    summary.index.name = "strategy"
    return summary


def make(
    returns: pd.DataFrame | Mapping[str, Sequence[float]],
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
    risk_free_rate: float = 0.0,
    title: str | None = "Risk vs. Return",
) -> go.Figure:
    """Create a risk/return scatter plot."""

    summary = risk_return_summary(
        returns, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=summary["volatility"],
            y=summary["return"],
            mode="markers+text",
            text=summary.index,
            textposition="top center",
            marker=dict(size=10, color=summary["sharpe"], colorscale="Viridis", showscale=True),
            name="Strategies",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
    )

    return apply_theme(fig)


__all__ = ["DEFAULT_PERIODS_PER_YEAR", "risk_return_summary", "make"]
