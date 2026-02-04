"""Compute chart data for simulation outputs."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Mapping, Protocol, cast

import pandas as pd

from .. import configure_matplotlib_config_dir
from ..metrics import rolling as rolling_metrics

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    class _Pyplot(Protocol):
        def subplots(self, *args: object, **kwargs: object) -> tuple[Figure, Axes]: ...


@lru_cache(maxsize=1)
def _load_matplotlib() -> tuple["_Pyplot", type["Figure"]]:
    """Configure matplotlib and return pyplot + Figure class."""

    configure_matplotlib_config_dir()

    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    return cast("_Pyplot", plt), Figure


def _weights_to_frame(
    weights: Mapping[pd.Timestamp, pd.Series] | pd.DataFrame,
) -> pd.DataFrame:
    """Return weights history as a tidy DataFrame.

    The input can be a mapping of dates to weight Series or an already
    assembled DataFrame.  Missing values are filled with ``0.0`` and the
    index is sorted to ensure chronological order.
    """

    if isinstance(weights, pd.DataFrame):
        if weights.empty:
            raise ValueError("weights cannot be empty")
        return weights.sort_index().fillna(0.0)

    if not weights:
        raise ValueError("weights cannot be empty")

    return pd.DataFrame({d: s for d, s in weights.items()}).T.sort_index().fillna(0.0)


def equity_curve(returns: pd.Series) -> tuple[Figure, pd.DataFrame]:
    """Return equity curve figure and DataFrame from periodic returns."""

    if returns.empty:
        raise ValueError("returns cannot be empty")

    plt, _ = _load_matplotlib()
    eq_series: pd.Series = (1.0 + returns.fillna(0.0)).cumprod()
    curve: pd.DataFrame = eq_series.to_frame("equity")
    fig, ax = plt.subplots()
    curve.plot(ax=ax)
    ax.set_ylabel("Equity")
    fig.tight_layout()
    return fig, curve


def drawdown_curve(returns: pd.Series) -> tuple[Figure, pd.DataFrame]:
    """Return drawdown figure and DataFrame derived from ``returns``."""

    if returns.empty:
        raise ValueError("returns cannot be empty")

    plt, _ = _load_matplotlib()
    curve: pd.Series = (1.0 + returns.fillna(0.0)).cumprod()
    dd: pd.Series = curve / curve.cummax() - 1.0
    dd_df = dd.to_frame("drawdown")
    fig, ax = plt.subplots()
    dd_df.plot(ax=ax)
    ax.set_ylabel("Drawdown")
    fig.tight_layout()
    return fig, dd_df


def rolling_information_ratio(
    returns: pd.Series,
    benchmark: pd.Series | float | None = None,
    window: int = 12,
) -> tuple[Figure, pd.DataFrame]:
    """Rolling information ratio over ``window`` periods."""
    if returns.empty:
        raise ValueError("returns cannot be empty")

    plt, _ = _load_matplotlib()
    ir_series: pd.Series = rolling_metrics.rolling_information_ratio(returns, benchmark, window)
    ir_df: pd.DataFrame = ir_series.to_frame("rolling_ir")
    fig, ax = plt.subplots()
    ir_df.plot(ax=ax)
    ax.set_ylabel("Rolling IR")
    fig.tight_layout()
    return fig, ir_df


def turnover_series(
    weights: Mapping[pd.Timestamp, pd.Series] | pd.DataFrame,
) -> tuple[Figure, pd.DataFrame]:
    """Compute turnover figure and DataFrame from weights history."""

    w_df = _weights_to_frame(weights)
    plt, _ = _load_matplotlib()
    to = w_df.diff().abs().sum(axis=1).to_frame("turnover")
    fig, ax = plt.subplots()
    to.plot(ax=ax)
    ax.set_ylabel("Turnover")
    fig.tight_layout()
    return fig, to


def weights_heatmap(
    weights: Mapping[pd.Timestamp, pd.Series] | pd.DataFrame,
) -> tuple[Figure, pd.DataFrame]:
    """Return heatmap figure and DataFrame of portfolio weights."""

    w_df = _weights_to_frame(weights)
    plt, _ = _load_matplotlib()
    fig, ax = plt.subplots()
    cax = ax.imshow(w_df.T.values, aspect="auto", interpolation="none", origin="lower")
    ax.set_yticks(range(len(w_df.columns)))
    ax.set_yticklabels(w_df.columns)
    ax.set_xticks(range(len(w_df.index)))
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in w_df.index], rotation=90)
    fig.colorbar(cax, ax=ax, label="Weight")
    fig.tight_layout()
    return fig, w_df


def weights_heatmap_data(
    weights: Mapping[pd.Timestamp, pd.Series] | pd.DataFrame,
) -> pd.DataFrame:
    """Return DataFrame suitable for a weights heatmap.

    This function provides a clean interface for getting weights data
    ready for visualization. It directly calls the optimized internal
    helper to avoid creating unnecessary matplotlib figures.

    Args:
        weights: Mapping of dates to weight Series or DataFrame of weights

    Returns:
        DataFrame with dates as index and assets as columns, filled with 0.0
        for missing values and sorted chronologically.
    """
    return _weights_to_frame(weights)
