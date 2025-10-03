"""Bootstrap utilities for backtest equity curves."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .harness import BacktestResult


def _init_rng(random_state: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _validate_inputs(result: BacktestResult, n: int, block: int) -> pd.Series:
    if n < 1:
        raise ValueError("n must be at least 1")
    if block <= 0:
        raise ValueError("block must be a positive integer")

    if not isinstance(result.returns, pd.Series):
        raise TypeError("result.returns must be a pandas Series")
    if not isinstance(result.equity_curve, pd.Series):
        raise TypeError("result.equity_curve must be a pandas Series")

    realised = result.returns.dropna()
    if realised.empty:
        raise ValueError("result.returns must contain at least one non-NaN value")
    return realised.astype(float)


def _bootstrap_paths(
    returns: pd.Series,
    *,
    n_paths: int,
    block: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = returns.to_numpy(dtype=float, copy=False)
    periods = len(values)
    out = np.empty((n_paths, periods), dtype=float)
    for i in range(n_paths):
        t = 0
        while t < periods:
            start = int(rng.integers(periods))
            seg_idx = np.arange(start, start + block)
            seg = np.take(values, seg_idx, mode="wrap")
            seg_len = min(block, periods - t)
            out[i, t : t + seg_len] = seg[:seg_len]
            t += seg_len
    return out


def bootstrap_equity(
    result: BacktestResult,
    n: int = 500,
    block: int = 20,
    *,
    random_state: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """Estimate equity-curve uncertainty using block bootstrap sampling.

    Parameters
    ----------
    result:
        Backtest artefacts containing realised returns and equity curve.
    n:
        Number of bootstrap samples (paths) to draw.  Must be at least 1.
    block:
        Block length for the circular bootstrap expressed in periods.
    random_state:
        Optional seed or generator for deterministic sampling in tests.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed like ``result.equity_curve`` with columns
        ``p05``, ``median`` and ``p95`` representing the 5th percentile,
        median, and 95th percentile equity paths respectively.
    """

    realised = _validate_inputs(result, n=n, block=block)
    rng = _init_rng(random_state)

    sampled = _bootstrap_paths(realised, n_paths=n, block=block, rng=rng)
    equity_paths = np.cumprod(1.0 + sampled, axis=1)

    first_active = realised.index[0]
    first_return = float(realised.iloc[0])
    try:
        equity_after_first = float(result.equity_curve.loc[first_active])
    except KeyError:  # pragma: no cover - defensive alignment fallback
        eq_non_na = result.equity_curve.dropna()
        equity_after_first = float(eq_non_na.iloc[0]) if not eq_non_na.empty else 1.0
    if not np.isfinite(equity_after_first):
        equity_after_first = 1.0

    denom = 1.0 + first_return
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        base_value = 1.0
    else:
        base_value = equity_after_first / denom
        if not np.isfinite(base_value):
            base_value = 1.0

    equity_paths *= base_value

    quantiles = np.percentile(equity_paths, [5, 50, 95], axis=0)
    index = realised.index
    data = pd.DataFrame(
        {
            "p05": quantiles[0],
            "median": quantiles[1],
            "p95": quantiles[2],
        },
        index=index,
    )

    full_index = result.equity_curve.index
    band = data.reindex(full_index)
    if len(full_index) > len(index):
        # Preserve NaNs for the pre-live window so charts align with the
        # published equity curve.
        band.loc[~result.returns.notna(), :] = np.nan
    return band
