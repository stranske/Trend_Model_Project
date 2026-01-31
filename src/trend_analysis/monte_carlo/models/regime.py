"""Regime detection and regime-conditioned bootstrap sampling."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .base import (
    PricePathResult,
    apply_missingness_mask,
    log_returns_to_prices,
    normalize_frequency_code,
    normalize_price_frequency,
    prices_to_log_returns,
)
from .bootstrap import (
    _build_multi_columns,
    _build_simulation_index,
    _coerce_calibration_window,
    _coerce_mean_block_len,
    _last_valid_prices,
    _stationary_bootstrap_indices,
)

__all__ = ["RegimeLabeler", "RegimeConditionedBootstrapModel"]


class RegimeLabeler:
    """Label historical observations into calm/stress regimes."""

    def __init__(
        self,
        *,
        proxy_column: str,
        threshold_percentile: float = 75.0,
        lookback: int = 20,
        calm_label: str = "calm",
        stress_label: str = "stress",
    ) -> None:
        proxy = str(proxy_column).strip() if proxy_column is not None else ""
        if not proxy:
            raise ValueError("proxy_column must be provided")
        threshold = float(threshold_percentile)
        if not 0.0 <= threshold <= 100.0:
            raise ValueError("threshold_percentile must be between 0 and 100")
        lookback_value = int(lookback)
        if lookback_value < 1:
            raise ValueError("lookback must be >= 1")
        self.proxy_column = proxy
        self.threshold_percentile = threshold
        self.lookback = lookback_value
        self.calm_label = str(calm_label) or "calm"
        self.stress_label = str(stress_label) or "stress"
        self._labels: pd.Series | None = None
        self._threshold: float | None = None

    def fit(self, prices: pd.DataFrame) -> "RegimeLabeler":
        if prices.empty:
            raise ValueError("prices must not be empty")
        if self.proxy_column not in prices.columns:
            raise KeyError(f"proxy column '{self.proxy_column}' not found in prices")
        normalized = prices.sort_index()
        log_returns = prices_to_log_returns(normalized)
        if not log_returns.empty and log_returns.iloc[0].isna().all():
            log_returns = log_returns.iloc[1:]
        proxy_returns = log_returns[self.proxy_column].astype(float)

        if proxy_returns.dropna().empty:
            self._labels = pd.Series(dtype="string")
            self._threshold = math.nan
            return self

        window = max(int(self.lookback), 1)
        window = min(window, len(proxy_returns))
        if window <= 1:
            vol = proxy_returns.abs()
        else:
            vol = proxy_returns.rolling(window).std(ddof=0)
            if vol.dropna().empty:
                vol = proxy_returns.abs()
        vol_clean = vol.dropna()
        if vol_clean.empty:
            self._labels = pd.Series(dtype="string")
            self._threshold = math.nan
            return self

        threshold = float(np.nanpercentile(vol_clean.to_numpy(), self.threshold_percentile))
        labels = pd.Series(self.calm_label, index=proxy_returns.index, dtype="string")
        labels.loc[vol >= threshold] = self.stress_label
        labels = labels.ffill().bfill()

        self._labels = labels
        self._threshold = threshold
        return self

    def get_labels(self) -> pd.Series:
        if self._labels is None:
            raise RuntimeError("RegimeLabeler must be fitted before accessing labels")
        return self._labels.copy()

    def get_transition_matrix(self) -> pd.DataFrame:
        labels = self.get_labels()
        if labels.empty:
            return pd.DataFrame(dtype=float)
        calm = self.calm_label
        stress = self.stress_label
        unique = pd.unique(labels)
        regimes = [label for label in (calm, stress) if label in unique]
        if not regimes:
            regimes = list(unique)

        current = labels.iloc[:-1]
        nxt = labels.iloc[1:]
        counts = pd.crosstab(current, nxt).reindex(index=regimes, columns=regimes, fill_value=0)
        matrix = counts.astype(float)
        for regime in regimes:
            row_sum = float(matrix.loc[regime].sum())
            if row_sum <= 0:
                matrix.loc[regime, :] = 0.0
                matrix.loc[regime, regime] = 1.0
            else:
                matrix.loc[regime, :] = matrix.loc[regime, :] / row_sum
        return matrix


def _simulate_regime_path(
    *,
    n_periods: int,
    n_paths: int,
    transition: np.ndarray,
    initial_probs: np.ndarray,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    n_regimes = transition.shape[0]
    regimes = np.empty((n_paths, n_periods), dtype=int)
    regimes[:, 0] = rng.choice(n_regimes, size=n_paths, p=initial_probs)
    for t in range(1, n_periods):
        for path in range(n_paths):
            prev = regimes[path, t - 1]
            regimes[path, t] = rng.choice(n_regimes, p=transition[prev])
    return regimes


def _regime_conditioned_indices(
    *,
    n_obs: int,
    n_periods: int,
    n_paths: int,
    mean_block_len: float,
    regime_path: NDArray[np.int64],
    regime_buckets: dict[int, NDArray[np.int64]],
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    if n_obs <= 0:
        raise ValueError("n_obs must be >= 1")
    if n_periods <= 0:
        raise ValueError("n_periods must be >= 1")
    if n_paths <= 0:
        raise ValueError("n_paths must be >= 1")
    if mean_block_len <= 0:
        raise ValueError("mean_block_len must be > 0")

    p = 1.0 / mean_block_len
    new_block = rng.random((n_paths, n_periods)) < p
    new_block[:, 0] = True

    indices = np.empty((n_paths, n_periods), dtype=int)
    all_indices = np.arange(n_obs, dtype=int)

    for path in range(n_paths):
        current_pos = 0
        current_bucket = all_indices
        for t in range(n_periods):
            regime = int(regime_path[path, t])
            bucket = regime_buckets.get(regime, all_indices)
            if bucket.size == 0:
                bucket = all_indices
            if t == 0 or new_block[path, t] or regime != regime_path[path, t - 1]:
                current_pos = int(rng.integers(0, bucket.size))
                current_bucket = bucket
            else:
                if not np.shares_memory(bucket, current_bucket):
                    current_pos = int(rng.integers(0, bucket.size))
                    current_bucket = bucket
                else:
                    current_pos = (current_pos + 1) % bucket.size
            indices[path, t] = bucket[current_pos]
    return indices


class RegimeConditionedBootstrapModel:
    """Bootstrap log returns conditioned on a simulated regime path."""

    def __init__(
        self,
        *,
        mean_block_len: float | int = 6,
        calibration_window: int | None = None,
        frequency: str | None = None,
        regime_proxy_column: str | None = None,
        threshold_percentile: float = 75.0,
        lookback: int = 20,
    ) -> None:
        self.mean_block_len = _coerce_mean_block_len(mean_block_len)
        self.calibration_window = _coerce_calibration_window(calibration_window)
        self._frequency = normalize_frequency_code(frequency) if frequency else None
        self.regime_proxy_column = regime_proxy_column
        self.threshold_percentile = float(threshold_percentile)
        self.lookback = int(lookback)
        self._prices: pd.DataFrame | None = None
        self._log_returns: pd.DataFrame | None = None
        self._log_returns_values: NDArray[np.float64] | None = None
        self._missingness_mask: pd.DataFrame | None = None
        self._missingness_mask_values: NDArray[np.bool_[Any]] | None = None
        self._start_prices: pd.Series | None = None
        self._labels: pd.Series | None = None
        self._transition: pd.DataFrame | None = None
        self._regime_buckets: dict[int, NDArray[np.int64]] | None = None

    @property
    def frequency(self) -> str:
        return normalize_frequency_code(self._frequency)

    def fit(
        self, prices: pd.DataFrame, *, frequency: str | None = None
    ) -> "RegimeConditionedBootstrapModel":
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
        self._missingness_mask = log_returns.isna()
        self._missingness_mask_values = self._missingness_mask.to_numpy(dtype=bool)
        self._start_prices = _last_valid_prices(normalized)

        self._labels = None
        self._transition = None
        self._regime_buckets = None

        proxy = self.regime_proxy_column
        if proxy and proxy in normalized.columns:
            labeler = RegimeLabeler(
                proxy_column=proxy,
                threshold_percentile=self.threshold_percentile,
                lookback=self.lookback,
            ).fit(normalized)
            labels = labeler.get_labels().reindex(log_returns.index).ffill().bfill()
            if not labels.empty and not labels.dropna().empty:
                self._labels = labels
                self._transition = labeler.get_transition_matrix()
                self._regime_buckets = {}
                label_values = labels.to_numpy()
                for idx, label in enumerate(self._transition.index):
                    self._regime_buckets[idx] = np.flatnonzero(label_values == label)
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
            or self._missingness_mask is None
            or self._missingness_mask_values is None
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

        if self._labels is None or self._transition is None or self._regime_buckets is None:
            indices = _stationary_bootstrap_indices(
                n_obs=n_obs,
                n_periods=n_periods,
                n_paths=n_paths,
                mean_block_len=self.mean_block_len,
                rng=rng,
            )
        else:
            transition = self._transition.to_numpy(dtype=float)
            regimes = list(self._transition.index)
            if self._regime_buckets is None:
                label_values = self._labels.to_numpy()
                buckets = {
                    idx: np.flatnonzero(label_values == label) for idx, label in enumerate(regimes)
                }
            else:
                buckets = {
                    idx: self._regime_buckets.get(idx, np.array([], dtype=int))
                    for idx in range(len(regimes))
                }
            counts = np.array([bucket.size for bucket in buckets.values()], dtype=float)
            total = counts.sum()
            if total <= 0:
                initial_probs = np.full(len(regimes), 1.0 / len(regimes))
            else:
                initial_probs = counts / total
            regime_path = _simulate_regime_path(
                n_periods=n_periods,
                n_paths=n_paths,
                transition=transition,
                initial_probs=initial_probs,
                rng=rng,
            )
            indices = _regime_conditioned_indices(
                n_obs=n_obs,
                n_periods=n_periods,
                n_paths=n_paths,
                mean_block_len=self.mean_block_len,
                regime_path=regime_path,
                regime_buckets=buckets,
                rng=rng,
            )

        sampled = data[indices]
        missingness = self._missingness_mask_values[indices]
        sampled = np.swapaxes(sampled, 0, 1)
        missingness = np.swapaxes(missingness, 0, 1)

        columns = _build_multi_columns(self._log_returns.columns, n_paths)
        log_return_frame = pd.DataFrame(
            sampled.reshape(n_periods, n_paths * n_assets), index=index, columns=columns
        )
        mask = pd.DataFrame(
            missingness.reshape(n_periods, n_paths * n_assets),
            index=index,
            columns=columns,
        )
        log_return_frame = log_return_frame.mask(mask)
        mask = mask | log_return_frame.isna()

        start_series = pd.Series(
            np.tile(self._start_prices.to_numpy(), n_paths), index=columns, dtype=float
        )
        prices = log_returns_to_prices(log_return_frame, start_series)
        prices = apply_missingness_mask(prices, mask)

        return PricePathResult(
            prices=prices,
            log_returns=log_return_frame,
            missingness_mask=mask,
            frequency=freq,
            start_date=index[0],
        )
