"""Walk‑forward cross‑validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Callable, Mapping, Any, cast

import numpy as np

import pandas as pd


@dataclass
class Split:
    """Container describing a single walk‑forward split."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex


@dataclass
class WalkForwardResult:
    """Result bundle returned by :func:`walk_forward`."""

    splits: List[Split]
    full: pd.Series
    oos: pd.Series
    by_regime: pd.DataFrame


def _prepare_index(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex or 'Date' column")
    return df.sort_index()


def _generate_splits(
    index: pd.DatetimeIndex, train: int, test: int, step: int
) -> List[Split]:
    splits: List[Split] = []
    start = 0
    n = len(index)
    while start + train + test <= n:
        train_idx = index[start : start + train]
        test_idx = index[start + train : start + train + test]
        splits.append(
            Split(
                train_start=train_idx[0],
                train_end=train_idx[-1],
                test_start=test_idx[0],
                test_end=test_idx[-1],
                train_index=train_idx,
                test_index=test_idx,
            )
        )
        start += step
    return splits


def walk_forward(
    df: pd.DataFrame,
    *,
    train_size: int,
    test_size: int,
    step_size: int,
    metric_cols: Sequence[str] | None = None,
    regimes: pd.Series | None = None,
    agg: (
        Callable[..., Any]
        | str
        | np.ufunc
        | Mapping[Any, Callable[..., Any] | str | np.ufunc]
        | List[str]
    ) = "mean",
) -> WalkForwardResult:
    """Run a simple walk‑forward aggregation.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with a ``Date`` column or ``DatetimeIndex``.
    train_size, test_size, step_size : int
        Number of rows for train, test and step respectively.
    metric_cols : sequence of str, optional
        Columns to aggregate; defaults to all non-date columns.
    regimes : pd.Series, optional
        Regime labels indexed by date.  Only test windows are aggregated by
        regime.  Missing labels are ignored.
    agg : str or iterable of str, default 'mean'
        Aggregation to apply via :meth:`pandas.DataFrame.agg`.

    Returns
    -------
    WalkForwardResult
        Structure containing splits and aggregated metrics.
    """

    df = _prepare_index(df)
    metrics_df = df if metric_cols is None else df[list(metric_cols)]

    splits = _generate_splits(
        cast(pd.DatetimeIndex, metrics_df.index), train_size, test_size, step_size
    )

    if splits:
        oos_index: pd.DatetimeIndex = pd.DatetimeIndex([])
        for sp in splits:
            oos_index = pd.DatetimeIndex(oos_index.union(sp.test_index))
        oos_df = metrics_df.loc[oos_index]
        oos_metrics = oos_df.agg(cast(Any, agg))
    else:
        oos_df = metrics_df.iloc[0:0]
        oos_metrics = oos_df.agg(cast(Any, agg))

    full_metrics = metrics_df.agg(cast(Any, agg))

    by_regime = pd.DataFrame()
    if regimes is not None and not oos_df.empty:
        reg = regimes.reindex(metrics_df.index)
        by_regime = oos_df.groupby(reg.loc[oos_df.index]).agg(cast(Any, agg))

    return WalkForwardResult(
        splits=splits,
        full=full_metrics,
        oos=oos_metrics,
        by_regime=by_regime,
    )


__all__ = ["walk_forward", "Split", "WalkForwardResult"]
