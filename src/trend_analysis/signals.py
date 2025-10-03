"""Unified signal engine for trend strategies.

The module exposes a :func:`generate_signals` entry-point that produces a
:class:`SignalFrame` representing the intermediate transformation stages of the
trend signal pipeline.  The function supports a baseline time-series momentum
signal with optional volatility adjustment and cross-sectional z-score
normalisation.  The resulting frame enforces a consistent schema regardless of
which features are enabled and ensures execution lag is applied before the
signals reach the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

__all__ = ["TrendSpec", "SignalFrame", "generate_signals"]


@dataclass(frozen=True)
class TrendSpec:
    """Specification describing how trend signals should be constructed.

    Parameters
    ----------
    lookback:
        Number of periods used to compute the time-series momentum signal.
    vol_lookback:
        Optional rolling window for realised volatility.  When ``None`` the
        lookback used for momentum is re-used.
    use_vol_adjust:
        Divide the momentum signal by realised volatility when set to ``True``.
    use_zscore:
        Apply cross-sectional z-score normalisation when ``True``.
    execution_lag:
        Number of periods to shift the final signal forward to avoid
        look-ahead.  A value of ``0`` disables the lag.
    """

    lookback: int = 1
    vol_lookback: int | None = None
    use_vol_adjust: bool = False
    use_zscore: bool = False
    execution_lag: int = 1

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError("lookback must be at least 1")
        if self.vol_lookback is not None and self.vol_lookback < 1:
            raise ValueError("vol_lookback must be at least 1 when provided")
        if self.execution_lag < 0:
            raise ValueError("execution_lag cannot be negative")

    @property
    def effective_vol_window(self) -> int:
        """Return the window length used for volatility adjustment."""

        return self.vol_lookback or self.lookback


class SignalFrame:
    """Container representing the stages of the signal pipeline.

    The frame stores data in a :class:`pandas.DataFrame` with a two-level column
    index.  The first level identifies the transformation stage while the second
    level enumerates the asset symbols.  This structure guarantees callers can
    rely on a consistent schema independent of the configuration used to
    generate the signal.
    """

    _STAGES = ("raw_momentum", "vol_adjusted", "normalized", "signal")

    def __init__(self, frame: pd.DataFrame):
        if not isinstance(frame, pd.DataFrame):  # pragma: no cover - guardrail
            raise TypeError("frame must be a pandas DataFrame")
        if frame.columns.nlevels != 2:
            raise ValueError("SignalFrame requires a two-level column index")
        stages = frame.columns.get_level_values(0)
        missing = [stage for stage in self._STAGES if stage not in stages]
        if missing:
            raise ValueError(f"SignalFrame missing columns for stages: {missing}")
        assets = frame.columns.get_level_values(1).unique()
        target_columns = pd.MultiIndex.from_product(
            (self._STAGES, assets), names=frame.columns.names
        )
        self._frame = frame.reindex(columns=target_columns)

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - thin delegation
        return getattr(self._frame, item)

    def __getitem__(self, key) -> Any:  # pragma: no cover - thin delegation
        return self._frame.__getitem__(key)

    def __iter__(self) -> Iterator[str]:  # pragma: no cover - thin delegation
        return iter(self._frame)

    def __repr__(self) -> str:  # pragma: no cover - formatting helper
        return f"SignalFrame({repr(self._frame)})"

    @property
    def frame(self) -> pd.DataFrame:
        """Return a defensive copy of the underlying DataFrame."""

        return self._frame.copy()

    def stage(self, name: str) -> pd.DataFrame:
        """Return the DataFrame slice associated with a particular stage."""

        if name not in self._STAGES:
            raise KeyError(f"Unknown stage '{name}'")
        return self._frame.xs(name, axis=1, level=0)

    @property
    def final(self) -> pd.DataFrame:
        """Convenience accessor for the executed signal layer."""

        return self.stage("signal")


def generate_signals(
    prices: pd.DataFrame,
    spec: TrendSpec,
    *,
    rebalance_dates: Sequence[pd.Timestamp] | Iterable[pd.Timestamp] | None = None,
) -> SignalFrame:
    """Return trend signals derived from ``prices`` according to ``spec``.

    Parameters
    ----------
    prices:
        Historical asset prices with a monotonically increasing ``DatetimeIndex``.
    spec:
        Configuration object describing the requested signal transformations.
    rebalance_dates:
        Optional sequence of timestamps indicating when the strategy rebalances.
        Execution lag is applied on these dates.
    """

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex")
    if not prices.index.is_monotonic_increasing:
        prices = prices.sort_index()

    prices = prices.astype(float)
    assets = list(prices.columns)
    returns = prices.pct_change()
    raw_momentum = prices.pct_change(spec.lookback)

    if spec.use_vol_adjust:
        window = spec.effective_vol_window
        realised_vol = returns.rolling(window).std()
        realised_vol = realised_vol.replace(0.0, np.nan)
        vol_adjusted = raw_momentum.divide(realised_vol)
    else:
        vol_adjusted = raw_momentum

    if spec.use_zscore:
        normalized = _zscore_normalise(vol_adjusted)
    else:
        normalized = vol_adjusted

    executed = normalized
    if rebalance_dates is not None and spec.execution_lag > 0:
        executed = _apply_execution_lag(
            normalized,
            rebalance_dates=rebalance_dates,
            periods=spec.execution_lag,
        )

    frame = _assemble_frame(
        index=prices.index,
        assets=assets,
        stages={
            "raw_momentum": raw_momentum,
            "vol_adjusted": vol_adjusted,
            "normalized": normalized,
            "signal": executed,
        },
    )
    return SignalFrame(frame)


def _assemble_frame(
    *,
    index: pd.Index,
    assets: Sequence[str],
    stages: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        (SignalFrame._STAGES, assets), names=("stage", "asset")
    )
    combined = pd.DataFrame(index=index, columns=columns, dtype=float)
    idx = pd.IndexSlice
    for stage in SignalFrame._STAGES:
        data = stages[stage].reindex(index=index, columns=assets)
        combined.loc[:, idx[stage, :]] = data.to_numpy()
    return combined


def _zscore_normalise(frame: pd.DataFrame) -> pd.DataFrame:
    demeaned = frame.sub(frame.mean(axis=1, skipna=True), axis=0)
    std = frame.std(axis=1, skipna=True, ddof=0)
    std = std.replace(0.0, np.nan)
    normalized = demeaned.divide(std, axis=0)
    normalized = normalized.where(np.isfinite(normalized))
    # Where all entries were NaN we retain NaNs.  Where std was zero, fallback to
    # zeros (no cross-sectional differentiation) while preserving original NaNs.
    zero_std_rows = std.isna() & frame.notna().any(axis=1)
    if zero_std_rows.any():
        normalized.loc[zero_std_rows] = 0.0
        normalized = normalized.where(frame.notna())
    return normalized


def _apply_execution_lag(
    frame: pd.DataFrame,
    *,
    rebalance_dates: Sequence[pd.Timestamp] | Iterable[pd.Timestamp],
    periods: int,
) -> pd.DataFrame:
    if periods <= 0:
        return frame

    shifted = frame.shift(periods)
    rebal_index = frame.index.intersection(pd.DatetimeIndex(list(rebalance_dates)))
    if not len(rebal_index):
        return frame

    result = frame.copy()
    result.loc[rebal_index] = shifted.loc[rebal_index]
    return result
