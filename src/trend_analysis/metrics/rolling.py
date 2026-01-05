"""Rolling metrics for time-series analysis."""

from __future__ import annotations

import pandas as pd

from ..perf.rolling_cache import compute_dataset_hash, get_cache


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

    base_returns = returns
    if benchmark is None:
        bench = pd.Series(0.0, index=base_returns.index)
    elif isinstance(benchmark, pd.Series):
        bench = benchmark.reindex_like(base_returns).fillna(0.0)
    else:
        bench = pd.Series(float(benchmark), index=base_returns.index)

    cache = get_cache()

    def _compute() -> pd.Series:
        excess = base_returns - bench
        mean = excess.rolling(window).mean()
        std = excess.rolling(window).std(ddof=1)
        ir = mean / std.replace(0.0, pd.NA)
        return ir.rename("rolling_ir")

    if cache.is_enabled():
        dataset_hash = compute_dataset_hash([base_returns, bench])
        idx = base_returns.index
        if hasattr(idx, "freqstr") and idx.freqstr:
            freq = str(idx.freqstr)
        else:
            try:
                freq = pd.infer_freq(idx)
            except (ValueError, TypeError):
                freq = None
        freq_tag = freq or "unknown"
        method_tag = "rolling_information_ratio_ddof1"
        return cache.get_or_compute(dataset_hash, int(window), freq_tag, method_tag, _compute)

    return _compute()


__all__ = ["rolling_information_ratio"]
