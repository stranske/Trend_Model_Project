"""Walk‑forward cross‑validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from trend_analysis.metrics import information_ratio


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
    full: pd.DataFrame
    oos: pd.DataFrame
    by_regime: pd.DataFrame
    oos_windows: pd.DataFrame
    periods_per_year: int


def _prepare_index(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex or 'Date' column")
    return df.sort_index()


def _generate_splits(index: pd.DatetimeIndex, train: int, test: int, step: int) -> List[Split]:
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


def _to_dataframe(
    obj: pd.Series | pd.DataFrame, *, default_name: str | None = None
) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        df.index = df.index.map(str)
        return df
    if isinstance(obj, pd.Series):
        df = obj.to_frame().T
        name = obj.name if obj.name is not None else default_name or "value"
        df.index = [str(name)]
        return df
    raise TypeError(f"Unsupported aggregation result type: {type(obj)!r}")


def _flatten_agg_result(
    obj: pd.Series | pd.DataFrame, *, default_name: str | None = None
) -> pd.Series:
    df = _to_dataframe(obj, default_name=default_name)
    if df.empty:
        return pd.Series(dtype=float)
    flattened = df.T.stack(future_stack=True)
    flattened.index = pd.MultiIndex.from_tuples(flattened.index, names=["metric", "statistic"])
    return flattened


def _information_ratio_frame(
    ir: float | pd.Series,
    columns: pd.Index,
) -> pd.DataFrame:
    if isinstance(ir, pd.Series):
        ser = ir.reindex(columns)
    else:
        data = {col: np.nan for col in columns}
        if len(columns):
            data[columns[0]] = float(ir)
        ser = pd.Series(data)
    ser.name = "information_ratio"
    return ser.to_frame().T


def _infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 1

    diffs = np.diff(index.values.astype("datetime64[ns]").astype(np.int64))
    if len(diffs) == 0:
        return 1

    median_ns = np.median(diffs)
    if median_ns <= 0:
        return 1

    median_days = median_ns / (24 * 60 * 60 * 1e9)
    if median_days <= 0:
        return 1

    approx = int(round(365 / median_days))
    if approx >= 300:
        return 252
    if 45 <= approx <= 60:
        return 52
    if 10 <= approx <= 14:
        return 12
    if 3 <= approx <= 5:
        return 4
    if approx <= 0:
        return 1
    return approx


def _agg_label(agg: Any) -> str:
    if isinstance(agg, str):
        return agg
    if callable(agg):
        name = getattr(agg, "__name__", None)
        if name:
            return str(name)
    return "value"


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

    periods_per_year = _infer_periods_per_year(cast(pd.DatetimeIndex, metrics_df.index))
    agg_label = _agg_label(agg)

    oos_index: pd.DatetimeIndex = pd.DatetimeIndex([])
    window_rows: List[dict[tuple[Any, Any], Any]] = []

    for sp in splits:
        test_df = metrics_df.loc[sp.test_index]
        oos_index = pd.DatetimeIndex(oos_index.union(sp.test_index))

        row: dict[tuple[Any, Any], Any] = {
            ("window", "train_start"): sp.train_start,
            ("window", "train_end"): sp.train_end,
            ("window", "test_start"): sp.test_start,
            ("window", "test_end"): sp.test_end,
            ("window", "train_count"): len(sp.train_index),
            ("window", "test_count"): len(sp.test_index),
        }

        agg_result = test_df.agg(cast(Any, agg))
        flattened = _flatten_agg_result(agg_result, default_name=agg_label)
        for key, value in flattened.items():
            row[(key[0], key[1])] = value

        if not test_df.empty:
            ir_vals = information_ratio(test_df, benchmark=0.0, periods_per_year=periods_per_year)
            if isinstance(ir_vals, pd.Series):
                for metric_name, ir_val in ir_vals.items():
                    row[(metric_name, "information_ratio")] = float(ir_val)
            else:
                columns = test_df.columns
                if len(columns):
                    row[(columns[0], "information_ratio")] = float(ir_vals)

        window_rows.append(row)

    oos_df = metrics_df.loc[oos_index] if len(oos_index) else metrics_df.iloc[0:0]

    full_metrics = _to_dataframe(metrics_df.agg(cast(Any, agg)), default_name=agg_label)
    full_ir = _information_ratio_frame(
        information_ratio(metrics_df, benchmark=0.0, periods_per_year=periods_per_year),
        metrics_df.columns,
    )
    full = pd.concat([full_metrics, full_ir], axis=0, sort=False)
    full.index.name = "statistic"

    oos_metrics = _to_dataframe(oos_df.agg(cast(Any, agg)), default_name=agg_label)
    oos_ir = _information_ratio_frame(
        information_ratio(oos_df, benchmark=0.0, periods_per_year=periods_per_year),
        metrics_df.columns,
    )
    oos = pd.concat([oos_metrics, oos_ir], axis=0, sort=False)
    oos.index.name = "statistic"

    by_regime = pd.DataFrame()
    if regimes is not None and not oos_df.empty:
        reg = regimes.reindex(metrics_df.index)
        grouped = oos_df.groupby(reg.loc[oos_df.index], dropna=True)
        rows: dict[Any, pd.Series] = {}
        for label, subset in grouped:
            agg_res = subset.agg(cast(Any, agg))
            agg_df = _to_dataframe(agg_res, default_name=agg_label)
            ir_df = _information_ratio_frame(
                information_ratio(subset, benchmark=0.0, periods_per_year=periods_per_year),
                subset.columns,
            )
            table = pd.concat([agg_df, ir_df], axis=0, sort=False)
            row_series = (
                table.T.stack(future_stack=True) if not table.empty else pd.Series(dtype=float)
            )
            if not row_series.empty:
                row_series.index = pd.MultiIndex.from_tuples(
                    row_series.index, names=["metric", "statistic"]
                )
            rows[label] = row_series
        if rows:
            by_regime = pd.DataFrame.from_dict(rows, orient="index")
            by_regime.index.name = "regime"
            by_regime.columns = pd.MultiIndex.from_tuples(
                by_regime.columns, names=["metric", "statistic"]
            )

    oos_windows = pd.DataFrame(window_rows)
    if not oos_windows.empty:
        oos_windows.index = pd.Index(range(1, len(oos_windows) + 1), name="window")
        oos_windows.columns = pd.MultiIndex.from_tuples(
            oos_windows.columns, names=["category", "statistic"]
        )
        window_cols = [col for col in oos_windows.columns if col[0] == "window"]
        metric_cols = [col for col in oos_windows.columns if col[0] != "window"]
        metric_cols.sort(key=lambda c: (str(c[0]), str(c[1])))
        oos_windows = oos_windows.loc[:, window_cols + metric_cols]

    return WalkForwardResult(
        splits=splits,
        full=full,
        oos=oos,
        by_regime=by_regime,
        oos_windows=oos_windows,
        periods_per_year=periods_per_year,
    )


__all__ = ["walk_forward", "Split", "WalkForwardResult"]
