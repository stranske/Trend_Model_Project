"""Stationary bootstrap return model."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from trend_analysis.timefreq import MONTHLY_DATE_FREQ

from .base import (
    PricePathResult,
    apply_missingness_mask,
    log_returns_to_prices,
    normalize_frequency_code,
    normalize_price_frequency,
    prices_to_log_returns,
)

__all__ = ["StationaryBootstrapModel"]


def _ensure_datetime_index(index: Iterable[object]) -> pd.DatetimeIndex:
    if isinstance(index, pd.DatetimeIndex):
        return index
    return pd.DatetimeIndex(index)


def _build_simulation_index(
    history_index: pd.DatetimeIndex,
    n_periods: int,
    frequency: str,
    start_date: pd.Timestamp | None,
) -> pd.DatetimeIndex:
    idx = _ensure_datetime_index(history_index)
    if start_date is None:
        last = idx.max()
        if frequency == "M":
            offset = pd.tseries.frequencies.to_offset(MONTHLY_DATE_FREQ)
        else:
            offset = pd.tseries.frequencies.to_offset("D")
        start = last + offset
    else:
        start = pd.Timestamp(start_date)
    if frequency == "M":
        return pd.date_range(start=start, periods=n_periods, freq=MONTHLY_DATE_FREQ)
    return pd.date_range(start=start, periods=n_periods, freq="D")


def _build_multi_columns(columns: pd.Index, n_paths: int) -> pd.MultiIndex:
    keys = list(range(n_paths))
    return pd.MultiIndex.from_product([keys, columns], names=["path", "asset"])


def _last_valid_prices(prices: pd.DataFrame) -> pd.Series:
    filled = prices.ffill()
    last = filled.iloc[-1]
    if last.isna().any():
        raise ValueError("prices must contain at least one non-missing value per asset")
    return last


def _coerce_mean_block_len(value: float | int) -> float:
    if isinstance(value, bool):
        raise ValueError("mean_block_len must be a positive number")
    try:
        mean = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("mean_block_len must be a positive number") from exc
    if not math.isfinite(mean) or mean <= 0:
        raise ValueError("mean_block_len must be a positive number")
    return mean


def _coerce_calibration_window(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("calibration_window must be an integer >= 2")
    try:
        window = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("calibration_window must be an integer >= 2") from exc
    if window < 2:
        raise ValueError("calibration_window must be an integer >= 2")
    return window


def _stationary_bootstrap_indices(
    n_obs: int,
    n_periods: int,
    n_paths: int,
    mean_block_len: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_obs <= 0:
        raise ValueError("n_obs must be >= 1")
    if n_periods <= 0:
        raise ValueError("n_periods must be >= 1")
    if n_paths <= 0:
        raise ValueError("n_paths must be >= 1")
    if mean_block_len <= 0:
        raise ValueError("mean_block_len must be > 0")

    p = 1.0 / mean_block_len
    starts = rng.integers(0, n_obs, size=(n_paths, n_periods))
    new_block = rng.random((n_paths, n_periods)) < p
    new_block[:, 0] = True

    indices = np.empty((n_paths, n_periods), dtype=int)
    current = starts[:, 0]
    indices[:, 0] = current
    for t in range(1, n_periods):
        start = starts[:, t]
        reset = new_block[:, t]
        current = np.where(reset, start, (current + 1) % n_obs)
        indices[:, t] = current
    return indices


class StationaryBootstrapModel:
    """Generate returns with a multivariate stationary bootstrap."""

    def __init__(
        self,
        *,
        mean_block_len: float | int = 6,
        calibration_window: int | None = None,
        frequency: str | None = None,
    ) -> None:
        self.mean_block_len = _coerce_mean_block_len(mean_block_len)
        self.calibration_window = _coerce_calibration_window(calibration_window)
        self._frequency = normalize_frequency_code(frequency) if frequency else None
        self._prices: pd.DataFrame | None = None
        self._log_returns: pd.DataFrame | None = None
        self._log_returns_values: np.ndarray | None = None
        self._start_prices: pd.Series | None = None

    @property
    def frequency(self) -> str:
        return normalize_frequency_code(self._frequency)

    def fit(
        self, prices: pd.DataFrame, *, frequency: str | None = None
    ) -> "StationaryBootstrapModel":
        if prices.empty:
            raise ValueError("prices must not be empty")
        target = normalize_frequency_code(frequency or self._frequency)
        normalized, _summary = normalize_price_frequency(prices, target)
        if self.calibration_window is not None:
            normalized = normalized.tail(self.calibration_window)
        normalized = normalized.sort_index()
        log_returns = prices_to_log_returns(normalized)
        if not log_returns.empty and log_returns.iloc[0].isna().all():
            log_returns = log_returns.iloc[1:]
        if log_returns.empty or log_returns.dropna(how="all").empty:
            raise ValueError("prices must contain at least two usable observations")

        self._frequency = target
        self._prices = normalized
        self._log_returns = log_returns
        self._log_returns_values = log_returns.to_numpy()
        self._start_prices = _last_valid_prices(normalized)
        return self

    def sample_prices(
        self,
        *,
        n_periods: int,
        n_paths: int,
        start_date: pd.Timestamp | None = None,
        frequency: str | None = None,
        seed: int | None = None,
    ) -> PricePathResult:
        if (
            self._prices is None
            or self._log_returns is None
            or self._log_returns_values is None
            or self._start_prices is None
        ):
            raise RuntimeError("Model must be fitted before sampling")
        if n_periods <= 0:
            raise ValueError("n_periods must be >= 1")
        if n_paths <= 0:
            raise ValueError("n_paths must be >= 1")

        freq = normalize_frequency_code(frequency or self._frequency)
        index = _build_simulation_index(self._prices.index, n_periods, freq, start_date)

        data = self._log_returns_values
        n_obs, n_assets = data.shape
        rng = np.random.default_rng(seed)
        indices = _stationary_bootstrap_indices(
            n_obs=n_obs,
            n_periods=n_periods,
            n_paths=n_paths,
            mean_block_len=self.mean_block_len,
            rng=rng,
        )
        sampled = data[indices]
        sampled = np.swapaxes(sampled, 0, 1)

        columns = _build_multi_columns(self._log_returns.columns, n_paths)
        log_return_frame = pd.DataFrame(
            sampled.reshape(n_periods, n_paths * n_assets), index=index, columns=columns
        )

        start_series = pd.Series(
            np.tile(self._start_prices.to_numpy(), n_paths), index=columns, dtype=float
        )
        prices = log_returns_to_prices(log_return_frame, start_series)
        mask = log_return_frame.isna()
        prices = apply_missingness_mask(prices, mask)

        return PricePathResult(
            prices=prices,
            log_returns=log_return_frame,
            missingness_mask=mask,
            frequency=freq,
            start_date=index[0],
        )
