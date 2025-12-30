"""Turnover and transaction cost utilities.

This module provides helper functions for computing realized turnover and
associated transaction costs from a weight history.  The implementation is
vectorised and free of plotting dependencies so it can be used from the metrics
package without requiring the ``viz`` helpers.
"""

from __future__ import annotations

from collections.abc import Mapping

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
    return pd.DataFrame({d: s for d, s in weights.items()}).T.sort_index().fillna(0.0)


def realized_turnover(
    weights: Mapping[pd.Timestamp, pd.Series] | pd.DataFrame,
) -> pd.DataFrame:
    """Return a DataFrame of realized turnover per period."""
    w_df = _weights_to_frame(weights)
    to = w_df.diff().abs().sum(axis=1).to_frame("turnover")
    return to


def turnover_cost(
    weights: Mapping[pd.Timestamp, pd.Series] | pd.DataFrame,
    cost_bps: float,
) -> pd.Series:
    """Return a Series of transaction cost deductions per period.

    Parameters
    ----------
    weights : Mapping or DataFrame
        Weight history used to compute turnover.
    cost_bps : float
        Linear transaction cost in basis points applied to turnover.
    """
    turn_df = realized_turnover(weights)
    return turn_df["turnover"] * (cost_bps / 10000.0)


__all__ = ["realized_turnover", "turnover_cost"]
