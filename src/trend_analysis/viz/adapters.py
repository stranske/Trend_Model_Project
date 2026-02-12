"""Adapters that normalize Monte Carlo outputs for visualization modules.

This module provides a stable adapter layer between Monte Carlo outputs and
chart components that expect predictable DataFrame schemas.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from typing import Any, Callable, ParamSpec, TypeVar, cast

import numpy as np
import pandas as pd

from trend_analysis.monte_carlo.results import build_summary_frame

st: Any
try:
    import streamlit as st
except Exception:  # pragma: no cover - streamlit is optional outside app runtime
    st = None

P = ParamSpec("P")
R = TypeVar("R")
SUMMARY_REQUIRED_COLUMNS: tuple[str, ...] = ("fold_id", "fold_label", "strategy", "paths")
"""Required columns for ``make_summary`` outputs."""

SUMMARY_REQUIRED_DTYPES: dict[str, str] = {
    "fold_id": "Int64",
    "fold_label": "string",
    "strategy": "string",
    "paths": "Int64",
}
"""Required dtypes for fixed summary columns.

Dynamic metric columns are float-like and coerced with ``pd.to_numeric``.
"""

PATHS_INDEX_NAMES: tuple[str, str] = ("date", "path")
"""Canonical index levels for ``make_paths`` outputs."""

PATHS_REQUIRED_COLUMNS: tuple[str, ...] = ("nav",)
"""Required columns for canonical path outputs."""

PATHS_REQUIRED_DTYPES: dict[str, str] = {
    "nav": "float64",
}
"""Required dtypes for canonical path output columns."""

ROLLING_REQUIRED_COLUMNS: tuple[str, ...] = (
    "rolling_mean",
    "rolling_std",
    "rolling_sharpe",
)
"""Required columns for ``rolling_stats`` outputs."""

LOOKBACK_PERIODS_VALIDATION_MESSAGE = (
    "lookback_periods must be a positive integer, or an iterable containing at least one "
    "positive integer"
)
"""Controlled validation error message for ``terminal_returns`` lookback input."""

NO_VALID_LOOKBACK_PERIODS_MESSAGE = "No valid lookback_periods provided"
"""Controlled validation error when iterable normalization produces no valid lookbacks."""

CACHING_REQUIRED_UNAVAILABLE_MESSAGE = "Caching required but streamlit.cache_data is unavailable"
"""Error raised when runtime requires caching but streamlit.cache_data cannot be used."""


def _cache_data(*args: object, **kwargs: object) -> Callable[[Callable[P, R]], Callable[P, R]]:
    cache_data = getattr(st, "cache_data", None) if st is not None else None
    if callable(cache_data):
        return cast(Callable[[Callable[P, R]], Callable[P, R]], cache_data(*args, **kwargs))
    if _is_caching_required():
        raise RuntimeError(CACHING_REQUIRED_UNAVAILABLE_MESSAGE)

    def _identity(func: Callable[P, R]) -> Callable[P, R]:
        return func

    return _identity


def _is_caching_required() -> bool:
    if os.environ.get("TREND_VIZ_REQUIRE_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return os.environ.get("TREND_ENV", "").strip().lower() in {"production", "prod"}


@_cache_data(show_spinner=False)
def _make_summary_cached(
    results_frame: pd.DataFrame,
    *,
    fold_selection: int | str | Sequence[int | str] | None = None,
) -> pd.DataFrame:
    if "strategy" not in results_frame.columns:
        raise ValueError("results_frame must include a 'strategy' column")

    pooled = _is_pooled_selection(fold_selection)
    filtered = _apply_fold_selection(results_frame, fold_selection)

    if pooled:
        no_fold = filtered.drop(
            columns=[col for col in ("fold_id", "fold_label") if col in filtered]
        )
        summary = build_summary_frame(no_fold)
    else:
        summary = build_summary_frame(filtered)

    return _normalize_summary_schema(summary)


@_cache_data(show_spinner=False)
def _make_paths_cached(nav_paths: pd.DataFrame) -> pd.DataFrame:
    frame = _normalize_nav_paths(nav_paths)
    if frame.empty:
        return pd.DataFrame(
            {"nav": pd.Series(dtype="float64")},
            index=pd.MultiIndex.from_arrays(
                [pd.DatetimeIndex([], name="date"), pd.Index([], name="path")],
                names=list(PATHS_INDEX_NAMES),
            ),
        )

    long = frame.stack(future_stack=True).rename("nav").to_frame()
    long.index.set_names(list(PATHS_INDEX_NAMES), inplace=True)
    long["nav"] = pd.to_numeric(long["nav"], errors="coerce").astype("float64")
    long = long[~long.index.duplicated(keep="last")]
    return long.sort_index()


def make_summary(
    results_frame: pd.DataFrame,
    *,
    fold_selection: int | str | Sequence[int | str] | None = None,
) -> pd.DataFrame:
    """Convert per-path Monte Carlo results into a chart-ready summary frame.

    Parameters
    ----------
    results_frame:
        Monte Carlo per-path output with at least a ``strategy`` column plus one
        or more numeric metric columns. Optional fold columns are ``fold_id``
        and ``fold_label``.
    fold_selection:
        Optional fold selector:
        - ``None`` / ``"all"`` / ``"all folds"`` keeps all folds.
        - ``int`` or numeric ``str`` filters on ``fold_id``.
        - non-numeric ``str`` filters on ``fold_label``.
        - sequence of int/str applies OR filtering across ids/labels.
        - ``"pooled"`` aggregates across folds into one row per strategy.

    Returns
    -------
    pd.DataFrame
        Summary-like frame with required columns:
        ``fold_id`` (Int64), ``fold_label`` (string), ``strategy`` (string),
        ``paths`` (Int64), plus dynamic numeric metric columns as ``float64``.
    """

    if not isinstance(results_frame, pd.DataFrame):
        raise TypeError("results_frame must be a pandas DataFrame")
    return _make_summary_cached(results_frame, fold_selection=fold_selection)


def make_paths(nav_paths: pd.DataFrame) -> pd.DataFrame:
    """Convert Monte Carlo ``nav_paths`` into canonical long-form paths.

    Input requirements
    ------------------
    - ``nav_paths`` must be a ``pd.DataFrame``.
    - Index must be datetime-like and convertible by ``pd.to_datetime``.
    - Columns may be plain path ids, or a MultiIndex with a ``path`` level and
      optional ``asset`` level. If an ``asset`` level exists, only ``"NAV"``
      rows are retained.

    Output schema
    -------------
    - MultiIndex index: ``("date", "path")``.
    - Required columns: ``nav`` (float64).
    """

    return _make_paths_cached(nav_paths)


def terminal_returns(
    paths: pd.DataFrame,
    *,
    lookback_periods: int | Iterable[object] | None = None,
) -> pd.DataFrame:
    """Calculate terminal returns per path from canonical ``make_paths`` output.

    Parameters
    ----------
    paths:
        Canonical paths DataFrame from ``make_paths`` with index
        ``("date", "path")`` and column ``nav``.
    lookback_periods:
        Optional trailing window (in rows). If ``None``, uses first to last NAV
        over the full horizon. If provided as an integer, return is computed from
        ``t - lookback_periods`` to final ``t``.
        If provided as an iterable, invalid entries are filtered out to include
        only values where ``type(x) is int`` and ``x > 0``.
        For iterable inputs, the first normalized lookback that fits the available
        rows is used; if none fit, the maximum available lookback is used.
        Raises ``ValueError("No valid lookback_periods provided")`` if iterable
        normalization produces no valid lookbacks.

    Returns
    -------
    pd.DataFrame
        Index is ``path`` with columns:
        - ``terminal_return`` (float64)
        - ``lookback_periods`` (Int64)
    """

    wide = _paths_to_wide_nav(paths)
    if wide.empty:
        return pd.DataFrame(
            {
                "terminal_return": pd.Series(dtype="float64"),
                "lookback_periods": pd.Series(dtype="Int64"),
            }
        )

    normalized_lookbacks = _normalize_lookback_periods(lookback_periods)

    if normalized_lookbacks is None:
        base = wide.ffill().iloc[0]
        periods_used = max(len(wide.index) - 1, 0)
    else:
        max_lookback = max(len(wide.index) - 1, 0)
        selected_lookback = next(
            (value for value in normalized_lookbacks if value <= max_lookback),
            max_lookback,
        )
        base = wide.ffill().iloc[-(selected_lookback + 1)]
        periods_used = selected_lookback

    terminal = wide.ffill().iloc[-1]
    returns = (terminal / base) - 1.0
    out = pd.DataFrame({"terminal_return": pd.to_numeric(returns, errors="coerce")})
    out["lookback_periods"] = pd.Series(periods_used, index=out.index, dtype="Int64")
    out.index = out.index.rename("path")
    return out


def _normalize_lookback_periods(
    lookback_periods: int | Iterable[object] | None,
) -> list[int] | None:
    if lookback_periods is None:
        return None
    if type(lookback_periods) is int and lookback_periods > 0:
        return [lookback_periods]
    if isinstance(lookback_periods, (str, bytes)):
        raise ValueError(LOOKBACK_PERIODS_VALIDATION_MESSAGE)
    if isinstance(lookback_periods, Iterable):
        valid_periods = [value for value in lookback_periods if type(value) is int and value > 0]
        if valid_periods:
            return valid_periods
        raise ValueError(NO_VALID_LOOKBACK_PERIODS_MESSAGE)
    raise ValueError(LOOKBACK_PERIODS_VALIDATION_MESSAGE)


def rolling_stats(
    paths: pd.DataFrame,
    *,
    window: int = 12,
    periods_per_year: int = 12,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Compute rolling mean/std/Sharpe from canonical paths output.

    Parameters
    ----------
    paths:
        Canonical paths DataFrame from ``make_paths``.
    window:
        Rolling window length in rows, must be > 1.
    periods_per_year:
        Annualization factor used for Sharpe scaling, must be > 0.
    risk_free_rate:
        Annual risk-free rate used to excess-adjust rolling mean returns.

    Returns
    -------
    pd.DataFrame
        MultiIndex index ``("date", "path")`` with columns:
        ``rolling_mean``, ``rolling_std``, ``rolling_sharpe`` (all float64).
    """

    if window <= 1:
        raise ValueError("window must be > 1")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be > 0")

    wide = _paths_to_wide_nav(paths)
    if wide.empty:
        return pd.DataFrame(
            {col: pd.Series(dtype="float64") for col in ROLLING_REQUIRED_COLUMNS},
            index=pd.MultiIndex.from_arrays(
                [pd.DatetimeIndex([], name="date"), pd.Index([], name="path")],
                names=list(PATHS_INDEX_NAMES),
            ),
        )

    returns = wide.pct_change().replace([np.inf, -np.inf], np.nan)
    roll_mean = returns.rolling(window=window, min_periods=window).mean()
    roll_std = returns.rolling(window=window, min_periods=window).std(ddof=0)

    rf_period = float(risk_free_rate) / float(periods_per_year)
    excess = roll_mean - rf_period
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = excess / roll_std
    sharpe = sharpe * np.sqrt(float(periods_per_year))

    out = pd.concat(
        {
            "rolling_mean": roll_mean.stack(future_stack=True),
            "rolling_std": roll_std.stack(future_stack=True),
            "rolling_sharpe": sharpe.stack(future_stack=True),
        },
        axis=1,
    )
    out.index.set_names(list(PATHS_INDEX_NAMES), inplace=True)
    return out.astype("float64").sort_index()


def path_correlations(paths: pd.DataFrame, *, window: int | None = None) -> pd.DataFrame:
    """Compute cross-path return correlations from canonical paths output.

    Parameters
    ----------
    paths:
        Canonical paths DataFrame from ``make_paths``.
    window:
        Optional trailing return rows to include. If ``None``, all available
        return rows are used.

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix with path labels as both index and columns.
    """

    if window is not None and window <= 1:
        raise ValueError("window must be > 1 when provided")

    wide = _paths_to_wide_nav(paths)
    if wide.empty:
        return pd.DataFrame()

    returns = wide.pct_change().replace([np.inf, -np.inf], np.nan)
    if window is not None:
        returns = returns.tail(window)
    corr = returns.corr()
    return corr.sort_index(axis=0).sort_index(axis=1)


def _is_pooled_selection(fold_selection: int | str | Sequence[int | str] | None) -> bool:
    if isinstance(fold_selection, str):
        return fold_selection.strip().lower() == "pooled"
    return False


def _apply_fold_selection(
    results_frame: pd.DataFrame,
    fold_selection: int | str | Sequence[int | str] | None,
) -> pd.DataFrame:
    if fold_selection is None:
        return results_frame
    if isinstance(fold_selection, str):
        token = fold_selection.strip().lower()
        if token in {"all", "all folds", "pooled"}:
            return results_frame
        return _filter_single(results_frame, fold_selection)
    if isinstance(fold_selection, int):
        return _filter_single(results_frame, fold_selection)
    if isinstance(fold_selection, Sequence):
        if len(fold_selection) == 0:
            return results_frame.iloc[0:0]
        frames = [_filter_single(results_frame, item) for item in fold_selection]
        if not frames:
            return results_frame.iloc[0:0]
        selected = pd.concat(frames, axis=0).drop_duplicates()
        return selected
    raise TypeError("fold_selection must be None, int, str, or a sequence of int/str")


def _filter_single(results_frame: pd.DataFrame, fold_selector: int | str) -> pd.DataFrame:
    if isinstance(fold_selector, int):
        if "fold_id" not in results_frame.columns:
            raise ValueError("fold_selection by id requires 'fold_id' in results_frame")
        fold_ids = pd.to_numeric(results_frame["fold_id"], errors="coerce")
        return results_frame[fold_ids == int(fold_selector)]

    text = str(fold_selector).strip()
    if text.isdigit():
        if "fold_id" in results_frame.columns:
            fold_ids = pd.to_numeric(results_frame["fold_id"], errors="coerce")
            return results_frame[fold_ids == int(text)]
        raise ValueError("fold_selection by id requires 'fold_id' in results_frame")

    if "fold_label" not in results_frame.columns:
        raise ValueError("fold_selection by label requires 'fold_label' in results_frame")
    labels = results_frame["fold_label"].astype("string")
    return results_frame[labels == text]


def _normalize_summary_schema(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    for column in SUMMARY_REQUIRED_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame["fold_id"] = pd.to_numeric(frame["fold_id"], errors="coerce").astype("Int64")
    frame["fold_label"] = frame["fold_label"].astype("string")
    frame["strategy"] = frame["strategy"].astype("string")
    frame["paths"] = pd.to_numeric(frame["paths"], errors="coerce").fillna(0).astype("Int64")

    metric_cols = [col for col in frame.columns if col not in SUMMARY_REQUIRED_COLUMNS]
    for col in metric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")

    ordered_metrics = sorted(metric_cols)
    return frame[list(SUMMARY_REQUIRED_COLUMNS) + ordered_metrics]


def _normalize_nav_paths(nav_paths: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(nav_paths, pd.DataFrame):
        raise TypeError("nav_paths must be a pandas DataFrame")

    frame = nav_paths.copy()
    if frame.empty:
        return frame

    date_index = pd.to_datetime(frame.index, errors="coerce")
    if date_index.isna().any():
        raise ValueError("nav_paths index must be datetime-like")
    frame.index = pd.DatetimeIndex(date_index, name="date")

    if isinstance(frame.columns, pd.MultiIndex):
        names = [name or "" for name in frame.columns.names]
        path_level = names.index("path") if "path" in names else 0
        if "asset" in names:
            asset_level = names.index("asset")
            asset_values = frame.columns.get_level_values(asset_level)
            nav_mask = asset_values.astype("string") == "NAV"
            if not bool(nav_mask.any()):
                raise ValueError("nav_paths with an 'asset' level must include 'NAV'")
            frame = frame.loc[:, nav_mask]
        frame.columns = frame.columns.get_level_values(path_level)

    frame.columns = pd.Index(frame.columns, name="path")
    if frame.columns.duplicated().any():
        # Use transpose-groupby-transpose for pandas compatibility across versions
        # that deprecate/alter ``groupby(..., axis=1)`` behavior.
        frame = frame.T.groupby(level=0).mean().T
        frame.columns.name = "path"

    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def _paths_to_wide_nav(paths: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(paths, pd.DataFrame):
        raise TypeError("paths must be a pandas DataFrame")
    if "nav" not in paths.columns:
        raise ValueError("paths must include a 'nav' column")

    if isinstance(paths.index, pd.MultiIndex):
        index_names = list(paths.index.names)
        if len(index_names) != 2:
            raise ValueError("paths index must have exactly two levels: date and path")
        if "date" not in index_names or "path" not in index_names:
            raise ValueError("paths index levels must include 'date' and 'path'")
        nav_series = pd.to_numeric(paths["nav"], errors="coerce")
        wide = nav_series.unstack("path")
    else:
        if {"date", "path", "nav"}.issubset(paths.columns):
            frame = paths.copy()
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            if frame["date"].isna().any():
                raise ValueError("paths['date'] must be datetime-like")
            frame["nav"] = pd.to_numeric(frame["nav"], errors="coerce")
            wide = frame.pivot(index="date", columns="path", values="nav")
        else:
            raise ValueError(
                "paths must have a MultiIndex ('date', 'path') or columns ['date', 'path', 'nav']"
            )

    wide.index = pd.to_datetime(wide.index, errors="coerce")
    if wide.index.isna().any():
        raise ValueError("paths date index must be datetime-like")
    return wide.sort_index()


__all__ = [
    "PATHS_INDEX_NAMES",
    "PATHS_REQUIRED_COLUMNS",
    "PATHS_REQUIRED_DTYPES",
    "ROLLING_REQUIRED_COLUMNS",
    "SUMMARY_REQUIRED_COLUMNS",
    "SUMMARY_REQUIRED_DTYPES",
    "make_paths",
    "make_summary",
    "path_correlations",
    "rolling_stats",
    "terminal_returns",
]
