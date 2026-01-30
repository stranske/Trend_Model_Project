"""Shared utilities for price/return conversion and missingness handling."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def price_availability_mask(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean mask of finite, positive prices."""

    values = _to_float_array(prices)
    mask = np.isfinite(values) & (values > 0)
    return pd.DataFrame(mask, index=prices.index, columns=prices.columns)


def returns_availability_mask(price_availability: pd.DataFrame) -> pd.DataFrame:
    """Return mask where both current and prior prices are available."""

    if price_availability.empty:
        return price_availability.copy()
    mask = price_availability & price_availability.shift(1)
    mask.iloc[0] = False
    return mask


def apply_availability_mask(frame: pd.DataFrame, availability: pd.DataFrame) -> pd.DataFrame:
    """Apply an availability mask, aligning it to the frame."""

    aligned = availability.reindex(index=frame.index, columns=frame.columns, fill_value=False)
    return frame.where(aligned)


def prices_to_log_returns(
    prices: pd.DataFrame,
    *,
    price_availability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert prices to log returns, preserving missingness.

    The returned log returns are masked so that any period lacking a current
    or prior price is set to NaN.
    """

    availability = price_availability if price_availability is not None else price_availability_mask(prices)
    log_prices = np.log(prices.where(availability))
    log_returns = log_prices.diff()
    return log_returns.where(returns_availability_mask(availability))


def log_returns_to_prices(
    log_returns: pd.DataFrame,
    start_prices: pd.Series | Mapping[str, float] | np.ndarray,
    *,
    price_availability: pd.DataFrame | None = None,
    start_at_first_row: bool = True,
) -> pd.DataFrame:
    """Reconstruct prices from log returns.

    Args:
        log_returns: Log return series with shape (n_steps, n_assets).
        start_prices: Prices aligned to ``log_returns.columns``.
        price_availability: Optional availability mask for prices.
        start_at_first_row: When True, the first row is set to ``start_prices``
            (useful when log returns include a leading NaN from ``diff``).
    """

    availability = _coerce_availability(log_returns, price_availability)
    start_series = _coerce_start_prices(start_prices, log_returns.columns)

    valid_start = np.isfinite(start_series) & (start_series > 0)
    availability = availability & valid_start

    prices = pd.DataFrame(index=log_returns.index, columns=log_returns.columns, dtype=float)
    current = start_series.copy()

    for idx, timestamp in enumerate(log_returns.index):
        if idx == 0 and start_at_first_row:
            row_availability = availability.iloc[0]
            prices.iloc[0] = start_series.where(row_availability, np.nan)
            current = current.where(~row_availability, prices.iloc[0])
            continue

        row = log_returns.iloc[idx]
        row_finite = pd.Series(np.isfinite(_to_float_array(row)), index=log_returns.columns)
        row_availability = availability.iloc[idx] & row_finite & np.isfinite(current)

        next_price = current * np.exp(row.fillna(0.0))
        prices.iloc[idx] = next_price.where(row_availability, np.nan)
        current = current.where(~row_availability, next_price)

    return prices


def _coerce_start_prices(
    start_prices: pd.Series | Mapping[str, float] | np.ndarray,
    columns: pd.Index,
) -> pd.Series:
    if isinstance(start_prices, pd.Series):
        series = start_prices.copy()
        series = series.reindex(columns)
        return series.astype(float)
    if isinstance(start_prices, Mapping):
        return pd.Series({col: float(start_prices.get(col, np.nan)) for col in columns})
    if isinstance(start_prices, np.ndarray):
        if start_prices.shape[0] != len(columns):
            raise ValueError("start_prices must align with log_returns columns")
        return pd.Series(start_prices.astype(float), index=columns)
    raise TypeError("start_prices must be a Series, mapping, or numpy array")


def _coerce_availability(
    log_returns: pd.DataFrame,
    availability: pd.DataFrame | None,
) -> pd.DataFrame:
    if availability is None:
        values = _to_float_array(log_returns)
        mask = np.isfinite(values)
        return pd.DataFrame(mask, index=log_returns.index, columns=log_returns.columns)
    return availability.reindex(index=log_returns.index, columns=log_returns.columns, fill_value=False)


def _to_float_array(frame: pd.DataFrame | pd.Series) -> np.ndarray:
    try:
        return frame.to_numpy(dtype=float, copy=False)
    except (TypeError, ValueError):
        return frame.astype(float).to_numpy(copy=False)


__all__ = [
    "apply_availability_mask",
    "log_returns_to_prices",
    "price_availability_mask",
    "prices_to_log_returns",
    "returns_availability_mask",
]
