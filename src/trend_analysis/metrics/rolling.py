"""Rolling metrics for time-series analysis."""

from __future__ import annotations

import pandas as pd


def rolling_information_ratio(
    returns: pd.Series,
    benchmark: pd.Series | float | None = None,
    window: int = 12,
) -> pd.Series:
    """Return rolling information ratio over ``window`` periods.

    Parameters
    ----------
    returns:
        Periodic returns of the strategy.
    benchmark:
        Optional benchmark returns or a constant benchmark value.  Missing
        values are treated as ``0.0``.
    window:
        Number of periods in the rolling window.

    Returns
    -------
    pd.Series
        Rolling information ratio named ``rolling_ir``.
    """

    if benchmark is None:
        bench = pd.Series(0.0, index=returns.index)
    elif isinstance(benchmark, pd.Series):
        bench = benchmark.reindex_like(returns).fillna(0.0)
    else:
        bench = pd.Series(float(benchmark), index=returns.index)

    excess = returns - bench
    mean = excess.rolling(window).mean()
    std = excess.rolling(window).std(ddof=1)
    ir = mean / std.replace(0.0, pd.NA)
    return ir.rename("rolling_ir")


__all__ = ["rolling_information_ratio"]
