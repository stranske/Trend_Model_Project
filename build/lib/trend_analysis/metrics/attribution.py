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

from typing import TYPE_CHECKING, Any, Iterable, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trend_analysis.constants import NUMERICAL_TOLERANCE_MEDIUM

__all__ = [
    "compute_contributions",
    "export_contributions",
    "plot_contributions",
]


def compute_contributions(
    signal_pnls: pd.DataFrame,
    rebalancing_pnl: pd.Series,
    *,
    tolerance: float = NUMERICAL_TOLERANCE_MEDIUM,
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
    # Cast to numpy arrays for explicit type clarity with np.allclose
    if not np.allclose(check.to_numpy(), contrib["total"].to_numpy(), atol=tolerance):
        raise ValueError("Contributions do not sum to total within tolerance")

    return contrib


def export_contributions(contrib: pd.DataFrame, path: str) -> None:
    """Save the contribution table to ``path`` as CSV."""

    contrib.to_csv(path, index=True)


if TYPE_CHECKING:  # for static typing of matplotlib axes without runtime dependency chain
    from matplotlib.axes import Axes as _Axes


def plot_contributions(
    contrib: pd.DataFrame,
    *,
    ax: "_Axes | None" = None,
    labels: Iterable[str] | None = None,
) -> "_Axes":
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

    # Determine the axes to plot on, handling accidental sequences at runtime
    if ax is None:
        _, plot_ax = plt.subplots()
    else:
        obj: Any = ax
        if hasattr(obj, "__getitem__") and not isinstance(obj, str):
            plot_ax = cast("_Axes", obj[0])
        else:
            plot_ax = cast("_Axes", obj)

    if labels is None:
        labels = [c for c in contrib.columns if c != "total"]

    contrib[labels].cumsum().plot(ax=plot_ax)
    plot_ax.set_ylabel("Cumulative contribution")
    plot_ax.set_xlabel("Time")
    plot_ax.legend(title="Component")
    return plot_ax
