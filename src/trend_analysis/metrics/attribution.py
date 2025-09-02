"""Performance attribution helpers.

This module provides simple utilities to decompose a strategy's
return into contributions from individual signals and from the
rebalancing effect.  The implementation is intentionally lightweight so
it can be used in both the live library and unit tests.

Functions
---------
compute_contributions
    Combine per‑signal and rebalancing PnL series and ensure the
    contributions sum to the total return.
export_contributions
    Persist the contributions table to a CSV file.
plot_contributions
    Visualise cumulative contributions over time.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "compute_contributions",
    "export_contributions",
    "plot_contributions",
]


def compute_contributions(
    signal_pnls: pd.DataFrame,
    rebalancing_pnl: pd.Series,
    *,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Return a contribution table for signals and rebalancing.

    Parameters
    ----------
    signal_pnls:
        DataFrame where each column contains the PnL attributable to a
        single signal.  Index represents the time axis.
    rebalancing_pnl:
        Series with the incremental PnL arising from portfolio
        rebalancing for each period.  Must share the same index as
        ``signal_pnls``.
    tolerance:
        Maximum allowed absolute difference between the summed
        contributions and the reported total.

    Returns
    -------
    DataFrame
        A table with one column per signal, plus ``"rebalancing"`` and
        ``"total"`` columns.  ``"total"`` is the row‑wise sum of all
        contributions.
    """

    if not signal_pnls.index.equals(rebalancing_pnl.index):
        raise ValueError("Indexes of signal_pnls and rebalancing_pnl must match")

    contrib = signal_pnls.copy()
    contrib["rebalancing"] = rebalancing_pnl
    contrib["total"] = contrib.sum(axis=1)

    # Validate totals – contributions must sum to the total return
    check = contrib.drop(columns="total").sum(axis=1)
    if not np.allclose(check, contrib["total"], atol=tolerance):
        raise ValueError("Contributions do not sum to total within tolerance")

    return contrib


def export_contributions(contrib: pd.DataFrame, path: str) -> None:
    """Save the contribution table to ``path`` as CSV."""

    contrib.to_csv(path, index=True)


def plot_contributions(
    contrib: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    labels: Iterable[str] | None = None,
) -> plt.Axes:
    """Plot cumulative contributions over time.

    Parameters
    ----------
    contrib:
        Contribution table as produced by :func:`compute_contributions`.
    ax:
        Optional matplotlib axes to draw on.  A new figure and axes are
        created if omitted.
    labels:
        Optional iterable of column labels to plot.  Defaults to all
        non‑total columns.

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the chart was drawn.  This allows callers to
        further customise or save the figure.
    """

    if ax is None:
        _, ax = plt.subplots()

    if labels is None:
        labels = [c for c in contrib.columns if c != "total"]

    contrib[labels].cumsum().plot(ax=ax)
    ax.set_ylabel("Cumulative contribution")
    ax.set_xlabel("Time")
    ax.legend(title="Component")
    return ax
