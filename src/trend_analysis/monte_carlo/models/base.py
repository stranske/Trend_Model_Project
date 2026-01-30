"""Shared Monte Carlo price path model utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from trend_analysis.timefreq import MONTHLY_DATE_FREQ
from trend_analysis.util.frequency import FrequencySummary, detect_frequency

__all__ = [
    "PricePathModel",
    "PricePathResult",
    "BootstrapPricePathModel",
    "prices_to_log_returns",
    "log_returns_to_prices",
    "normalize_price_frequency",
    "extract_missingness_mask",
    "build_missingness_mask",
    "expand_mask_for_paths",
    "apply_missingness_mask",
]

_SUPPORTED_FREQUENCIES = {"D", "M"}


@dataclass(frozen=True, slots=True)
class PricePathResult:
    """Container for simulated price paths and metadata."""

    prices: pd.DataFrame
    log_returns: pd.DataFrame
    missingness_mask: pd.DataFrame
    frequency: str
    start_date: pd.Timestamp


class PricePathModel(ABC):
    """Interface for Monte Carlo models that generate price paths."""

    @property
    @abstractmethod
    def frequency(self) -> str:
        """Return the simulation frequency code (e.g., ``"D"`` or ``"M"``)."""

    @abstractmethod
    def simulate(
        self,
        *,
        n_periods: int,
        n_paths: int,
        start_date: pd.Timestamp | None = None,
        frequency: str | None = None,
        seed: int | None = None,
    ) -> PricePathResult:
        """Generate Monte Carlo price paths.

        Parameters
        ----------
        n_periods:
            Number of periods to simulate.
        n_paths:
            Number of independent paths to generate.
        start_date:
            Optional starting date for the first simulated observation. When
            ``None``, the next period after the fitted data is used.
        frequency:
            Optional override for the simulation frequency (``"D"`` or ``"M"``).
        seed:
            Optional random seed for reproducibility.
        """


def _require_positive_prices(prices: pd.DataFrame) -> None:
    if ((prices <= 0) & prices.notna()).any().any():
        raise ValueError("Prices must be positive to compute log returns")


def _ensure_datetime_index(index: Iterable[object]) -> pd.DatetimeIndex:
    if isinstance(index, pd.DatetimeIndex):
        return index
    return pd.DatetimeIndex(index)


def _normalize_frequency_code(freq: str | None) -> str:
    if not freq:
        return "M"
    code = str(freq).upper()
    if code not in _SUPPORTED_FREQUENCIES:
        allowed = ", ".join(sorted(_SUPPORTED_FREQUENCIES))
        raise ValueError(f"Unsupported frequency '{code}'. Use {allowed}.")
    return code


def prices_to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert price levels to log returns.

    Missing observations remain ``NaN``; non-positive prices raise ``ValueError``.
    """

    if prices.empty:
        return prices.copy()
    _require_positive_prices(prices)
    return np.log(prices / prices.shift(1))


def log_returns_to_prices(
    log_returns: pd.DataFrame, start_prices: pd.Series
) -> pd.DataFrame:
    """Convert log returns into price levels.

    ``start_prices`` must be positive and aligned with ``log_returns`` columns.
    """

    if log_returns.empty:
        return log_returns.copy()
    if not isinstance(start_prices, pd.Series):
        raise TypeError("start_prices must be a pandas Series")
    if ((start_prices <= 0) & start_prices.notna()).any():
        raise ValueError("start_prices must be positive")
    if not log_returns.columns.equals(start_prices.index):
        start_prices = start_prices.reindex(log_returns.columns)
    if start_prices.isna().any():
        raise ValueError("start_prices must align with log_returns columns")
    returns = log_returns.fillna(0.0)
    prices = np.exp(returns.cumsum()) * start_prices
    return prices


def normalize_price_frequency(
    prices: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, FrequencySummary]:
    """Normalize price history to a target frequency (daily or monthly)."""

    if prices.empty:
        summary = detect_frequency([])
        return prices.copy(), summary
    target_code = _normalize_frequency_code(target)
    idx = _ensure_datetime_index(prices.index)
    prices = prices.copy()
    prices.index = idx
    summary = detect_frequency(idx)
    if target_code == summary.code:
        return prices, summary
    if target_code == "M":
        return prices.resample(MONTHLY_DATE_FREQ).last(), summary
    return prices.resample("D").ffill(), summary


def extract_missingness_mask(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean mask marking missing observations in ``prices``."""

    return prices.isna()


def build_missingness_mask(
    mask: pd.DataFrame, index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Project a historical missingness mask onto a new index."""

    if mask.empty:
        return pd.DataFrame(False, index=index, columns=mask.columns)
    values = mask.to_numpy(dtype=bool)
    repeat = int(np.ceil(len(index) / values.shape[0]))
    tiled = np.tile(values, (repeat, 1))[: len(index)]
    return pd.DataFrame(tiled, index=index, columns=mask.columns)


def expand_mask_for_paths(mask: pd.DataFrame, n_paths: int) -> pd.DataFrame:
    """Repeat an asset mask across multiple Monte Carlo paths."""

    if n_paths <= 0:
        raise ValueError("n_paths must be >= 1")
    if mask.empty:
        return mask.copy()
    keys = list(range(n_paths))
    return pd.concat([mask] * n_paths, axis=1, keys=keys, names=["path", "asset"])


def apply_missingness_mask(prices: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Apply a missingness mask to simulated prices."""

    aligned = mask.reindex(index=prices.index, columns=prices.columns, fill_value=False)
    return prices.mask(aligned)


class BootstrapPricePathModel(PricePathModel):
    """Bootstrap log returns from history to simulate price paths."""

    def __init__(self, prices: pd.DataFrame, *, frequency: str | None = None) -> None:
        if prices.empty:
            raise ValueError("prices must not be empty")
        target = _normalize_frequency_code(frequency)
        normalized, summary = normalize_price_frequency(prices, target)
        self._frequency = target
        self._frequency_summary = summary
        self._prices = normalized.sort_index()
        self._log_returns = prices_to_log_returns(self._prices)
        self._missingness_mask = extract_missingness_mask(self._prices)

    @property
    def frequency(self) -> str:
        return self._frequency

    def simulate(
        self,
        *,
        n_periods: int,
        n_paths: int,
        start_date: pd.Timestamp | None = None,
        frequency: str | None = None,
        seed: int | None = None,
    ) -> PricePathResult:
        if n_periods <= 0:
            raise ValueError("n_periods must be >= 1")
        if n_paths <= 0:
            raise ValueError("n_paths must be >= 1")
        freq = _normalize_frequency_code(frequency or self._frequency)
        index = _build_simulation_index(self._prices.index, n_periods, freq, start_date)

        history = self._log_returns.dropna(how="all")
        n_assets = self._prices.shape[1]
        rng = np.random.default_rng(seed)
        if history.empty:
            sampled = np.zeros((n_periods, n_paths, n_assets))
        else:
            data = history.to_numpy()
            idx = rng.integers(0, data.shape[0], size=(n_periods * n_paths))
            sampled = data[idx].reshape(n_periods, n_paths, n_assets)

        columns = _build_multi_columns(self._prices.columns, n_paths)
        log_return_frame = pd.DataFrame(
            sampled.reshape(n_periods, n_paths * n_assets), index=index, columns=columns
        )

        start_prices = _last_valid_prices(self._prices)
        start_series = pd.Series(
            np.tile(start_prices.to_numpy(), n_paths), index=columns, dtype=float
        )
        prices = log_returns_to_prices(log_return_frame, start_series)

        base_mask = build_missingness_mask(self._missingness_mask, index)
        mask = expand_mask_for_paths(base_mask, n_paths)
        prices = apply_missingness_mask(prices, mask)

        return PricePathResult(
            prices=prices,
            log_returns=log_return_frame,
            missingness_mask=mask,
            frequency=freq,
            start_date=index[0],
        )


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


def _last_valid_prices(prices: pd.DataFrame) -> pd.Series:
    filled = prices.ffill()
    last = filled.iloc[-1]
    if last.isna().any():
        raise ValueError("prices must contain at least one non-missing value per asset")
    return last


def _build_multi_columns(columns: pd.Index, n_paths: int) -> pd.MultiIndex:
    keys = list(range(n_paths))
    return pd.MultiIndex.from_product([keys, columns], names=["path", "asset"])
