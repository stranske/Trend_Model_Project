"""Weight policy helpers for handling invalid signals and warm-up windows."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


WeightPolicyMode = str


def _as_series(values: Mapping[str, float] | pd.Series | None) -> pd.Series:
    """Return a float Series aligned to mapping keys without modifying order."""

    if values is None:
        return pd.Series(dtype=float)
    if isinstance(values, pd.Series):
        return values.astype(float)
    return pd.Series(values, dtype=float)


def _invalid_mask(weights: pd.Series, signals: pd.Series | None) -> pd.Series:
    mask = ~np.isfinite(weights)
    if signals is not None:
        mask |= ~np.isfinite(signals)
    return mask


def apply_weight_policy(
    weights: Mapping[str, float] | pd.Series,
    signals: Mapping[str, float] | pd.Series | None = None,
    *,
    mode: WeightPolicyMode = "drop",
    min_assets: int = 1,
    previous: Mapping[str, float] | pd.Series | None = None,
) -> pd.Series:
    """Clean weights and handle invalid signals according to ``mode``.

    Parameters
    ----------
    weights:
        Candidate weights for the rebalance.
    signals:
        Signal snapshot aligned to ``weights``. Any non-finite entries are
        treated as invalid. When ``None``, only the weight values themselves are
        validated.
    mode:
        Policy to apply when encountering invalid signals/weights:

        - ``"drop"``: remove invalid assets then renormalise to 1.0.
        - ``"carry"``: replace invalid weights with the previous weight when
          available; otherwise drop them. Renormalises to 1.0.
        - ``"cash"``: set invalid weights to zero and leave the remaining
          weights unnormalised so the residual represents a cash buffer.
    min_assets:
        Minimum number of assets required after filtering. If the threshold
        isn't met and ``previous`` supplies enough valid weights, the previous
        weights are used instead. Otherwise the policy falls back to an empty
        (all-cash) portfolio.
    previous:
        Optional previous-period weights used for the carry and min-asset
        fallbacks.

    Returns
    -------
    pd.Series
        Sanitised weights obeying the chosen policy with no NaNs.
    """

    current = _as_series(weights)
    if current.empty:
        return current

    signals_series = _as_series(signals)
    if not signals_series.empty:
        signals_series = signals_series.reindex(current.index)
    else:
        signals_series = None

    mode = str(mode or "drop").lower()
    min_assets = max(int(min_assets or 0), 0)

    mask = _invalid_mask(current, signals_series)

    if mode == "carry":
        prev = _as_series(previous)
        filled = current.copy()
        filled.loc[mask] = prev.reindex(current.index).loc[mask]
        current = filled
        # Recompute mask after attempting to fill from previous
        mask = _invalid_mask(current, signals_series)
    elif mode == "cash":
        # Zero out invalid weights but keep remaining sums to preserve cash
        current = current.mask(mask, 0.0)
    else:  # drop (default)
        current = current.loc[~mask]

    # Final cleanup for any remaining non-finite entries
    current = current.replace([np.inf, -np.inf], np.nan).dropna()

    if len(current.index) < min_assets:
        prev = _as_series(previous).replace([np.inf, -np.inf], np.nan).dropna()
        if len(prev.index) >= min_assets:
            current = prev.reindex(prev.index)
        else:
            # No valid fallback – remain in cash
            return pd.Series(dtype=float)

    if current.empty:
        return current

    if mode == "cash":
        # Preserve any implicit cash buffer by leaving weights unnormalised
        current = current.clip(lower=0.0)
        return current

    total = float(current.sum())
    if total == 0.0:
        return current.fillna(0.0)
    normalised = (current / total).fillna(0.0)
    return normalised


__all__ = ["apply_weight_policy"]
