"""Adapters that normalize Monte Carlo outputs for visualization modules.

This module provides a stable adapter layer between Monte Carlo outputs and
chart components that expect predictable DataFrame schemas.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from trend_analysis.monte_carlo.results import build_summary_frame

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
    if "strategy" not in results_frame.columns:
        raise ValueError("results_frame must include a 'strategy' column")

    pooled = _is_pooled_selection(fold_selection)
    filtered = _apply_fold_selection(results_frame, fold_selection)

    if pooled:
        no_fold = filtered.drop(columns=[col for col in ("fold_id", "fold_label") if col in filtered])
        summary = build_summary_frame(no_fold)
    else:
        summary = build_summary_frame(filtered)

    return _normalize_summary_schema(summary)


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
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    ordered_metrics = sorted(metric_cols)
    return frame[list(SUMMARY_REQUIRED_COLUMNS) + ordered_metrics]


__all__ = ["SUMMARY_REQUIRED_COLUMNS", "SUMMARY_REQUIRED_DTYPES", "make_summary"]
