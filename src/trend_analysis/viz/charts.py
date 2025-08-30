"""Compute chart data for simulation outputs."""

from __future__ import annotations

from typing import Mapping

import pandas as pd


def _weights_to_frame(
    weights: Mapping[pd.Timestamp, pd.Series] | pd.DataFrame,
) -> pd.DataFrame:
    """Return weights history as a tidy DataFrame.

    The input can be a mapping of dates to weight Series or an already
    assembled DataFrame.  Missing values are filled with ``0.0`` and the
    index is sorted to ensure chronological order.
    """

    if isinstance(weights, pd.DataFrame):
        return weights.sort_index().fillna(0.0)
    return (
        pd.DataFrame({d: s for d, s in weights.items()})
        .T.sort_index()
        .fillna(0.0)
    )


def equity_curve(returns: pd.Series) -> pd.DataFrame:
    """Return equity curve DataFrame from periodic returns."""

    curve = (1.0 + returns.fillna(0.0)).cumprod()
    return curve.to_frame("equity")


def drawdown_curve(returns: pd.Series) -> pd.DataFrame:
    """Return drawdown series derived from ``returns``."""

    curve = (1.0 + returns.fillna(0.0)).cumprod()
    dd = curve / curve.cummax() - 1.0
    return dd.to_frame("drawdown")


def rolling_information_ratio(
    returns: pd.Series,
    benchmark: pd.Series | float | None = None,
    window: int = 12,
) -> pd.DataFrame:
    """Rolling information ratio over ``window`` periods."""

    if benchmark is None:
        bench = pd.Series(0.0, index=returns.index)
    elif isinstance(benchmark, pd.Series):
        bench = benchmark.reindex_like(returns).fillna(0.0)
    else:
        bench = pd.Series(float(benchmark), index=returns.index)

    excess = returns - bench
    mean = excess.rolling(window).mean()
    std = excess.rolling(window).std(ddof=0)
    ir = mean / std.replace(0.0, pd.NA)
    return ir.to_frame("rolling_ir")


def turnover_series(
    weights: Mapping[pd.Timestamp, pd.Series] | pd.DataFrame,
) -> pd.DataFrame:
    """Compute turnover series from weights history."""

    w_df = _weights_to_frame(weights)
    to = w_df.diff().abs().sum(axis=1)
    return to.to_frame("turnover")


def weights_heatmap_data(
    weights: Mapping[pd.Timestamp, pd.Series] | pd.DataFrame,
) -> pd.DataFrame:
    """Return DataFrame suitable for a weights heatmap."""

    return _weights_to_frame(weights)

