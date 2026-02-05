"""Result containers and export helpers for Monte Carlo simulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "MonteCarloPathError",
    "MonteCarloResults",
    "StrategyEvaluation",
    "build_cross_fold_summary_frame",
    "build_pooled_distribution_frame",
    "build_pooled_summary_frame",
    "build_results_frame",
    "build_summary_frame",
    "export_results",
]

RESULT_BASE_COLUMNS = (
    "fold_id",
    "fold_label",
    "path_id",
    "strategy",
    "path_hash",
    "seed",
    "metric_source",
)


@dataclass(frozen=True)
class StrategyEvaluation:
    """Single strategy evaluation for one Monte Carlo path."""

    fold_id: int | None
    path_id: int
    strategy_name: str
    metrics: Mapping[str, float]
    metric_source: str | None
    path_hash: str
    fold_label: str | None = None
    seed: int | None = None
    diagnostic: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class MonteCarloPathError:
    """Error record for a failed path evaluation."""

    fold_id: int | None
    path_id: int
    strategy_name: str | None
    error_type: str
    message: str
    fold_label: str | None = None


@dataclass(frozen=True)
class MonteCarloResults:
    """Container for Monte Carlo evaluation outputs."""

    mode: str
    evaluations: Sequence[StrategyEvaluation]
    errors: Sequence[MonteCarloPathError]
    results_frame: pd.DataFrame
    summary_frame: pd.DataFrame
    cross_fold_summary_frame: pd.DataFrame | None = None
    pooled_summary_frame: pd.DataFrame | None = None
    pooled_distribution_frame: pd.DataFrame | None = None
    metadata: Mapping[str, Any] | None = None


def build_results_frame(evaluations: Iterable[StrategyEvaluation]) -> pd.DataFrame:
    """Return a flat results table for all strategy evaluations."""

    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        row: dict[str, Any] = {
            "fold_id": (int(evaluation.fold_id) if evaluation.fold_id is not None else None),
            "fold_label": evaluation.fold_label,
            "path_id": int(evaluation.path_id),
            "strategy": evaluation.strategy_name,
            "path_hash": evaluation.path_hash,
            "seed": evaluation.seed,
            "metric_source": evaluation.metric_source,
        }
        row.update({str(k): float(v) for k, v in evaluation.metrics.items()})
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=list(RESULT_BASE_COLUMNS))
    frame = pd.DataFrame(rows)
    base_cols = [col for col in RESULT_BASE_COLUMNS if col in frame.columns]
    other_cols = [col for col in frame.columns if col not in base_cols]
    return frame[base_cols + other_cols]


def build_summary_frame(results_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results per strategy."""

    fold_group_cols: list[str] = []
    if "fold_id" in results_frame.columns:
        fold_group_cols.append("fold_id")
        if "fold_label" in results_frame.columns:
            fold_group_cols.append("fold_label")

    if results_frame.empty:
        base_cols = ["strategy", "paths"]
        if fold_group_cols:
            base_cols = [*fold_group_cols, "strategy", "paths"]
        return pd.DataFrame(columns=base_cols)
    metric_cols = _metric_columns(results_frame)
    if fold_group_cols:
        grouped = _coerce_metric_frame(results_frame, metric_cols).groupby(
            [*fold_group_cols, "strategy"],
            dropna=False,
        )
    else:
        grouped = _coerce_metric_frame(results_frame, metric_cols).groupby(
            "strategy",
            dropna=False,
        )
    summary = grouped[metric_cols].mean(numeric_only=True)
    summary["paths"] = grouped.size()
    return summary.reset_index()


def build_pooled_summary_frame(results_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results per strategy across all folds (pooled summary)."""

    if results_frame.empty:
        return pd.DataFrame(
            columns=[
                "scope",
                "pooled_scope",
                "fold_id",
                "fold_label",
                "strategy",
                "paths",
                "folds",
            ]
        )

    metric_cols = _metric_columns(results_frame)

    grouped = _coerce_metric_frame(results_frame, metric_cols).groupby(
        "strategy",
        dropna=False,
    )
    summary = grouped[metric_cols].mean(numeric_only=True)
    summary["paths"] = grouped.size()
    if "fold_id" in results_frame.columns:
        summary["folds"] = grouped["fold_id"].nunique(dropna=False)
    pooled = summary.reset_index()
    pooled.insert(0, "scope", "pooled")
    pooled.insert(1, "pooled_scope", "summary")
    pooled.insert(2, "fold_id", None)
    pooled.insert(3, "fold_label", None)
    return pooled


def build_pooled_distribution_frame(results_frame: pd.DataFrame) -> pd.DataFrame:
    """Return a pooled distribution table across all folds."""

    if results_frame.empty:
        return pd.DataFrame(columns=["scope", "pooled_scope", *RESULT_BASE_COLUMNS])
    pooled = results_frame.copy()
    pooled.insert(0, "scope", "pooled")
    pooled.insert(1, "pooled_scope", "distribution")
    return pooled


def build_cross_fold_summary_frame(results_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize fold-level results for cross-fold comparison."""

    if results_frame.empty or "fold_id" not in results_frame.columns:
        return pd.DataFrame(columns=["scope", "fold_id", "strategy", "folds"])

    fold_summary = build_summary_frame(results_frame)
    if fold_summary.empty:
        return pd.DataFrame(columns=["scope", "fold_id", "strategy", "folds"])

    metric_cols = _metric_columns(fold_summary, include_paths=True)
    grouped = _coerce_metric_frame(fold_summary, metric_cols).groupby(
        "strategy",
        dropna=False,
    )
    stats = grouped[metric_cols].agg(["mean", "std", "min", "max", "median"])
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    stats["folds"] = grouped.size()
    cross_fold = stats.reset_index()
    cross_fold.insert(0, "scope", "cross_fold")
    cross_fold.insert(1, "fold_id", None)
    return cross_fold


def _metric_columns(frame: pd.DataFrame, *, include_paths: bool = False) -> list[str]:
    excluded = set(RESULT_BASE_COLUMNS)
    excluded.add("folds")
    if not include_paths:
        excluded.add("paths")
    metric_cols: list[str] = []
    for col in frame.columns:
        if col in excluded:
            continue
        series = frame[col]
        if pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            metric_cols.append(col)
            continue
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            non_null = series.dropna()
            if non_null.empty:
                continue
            if non_null.map(lambda value: isinstance(value, (bool, np.bool_))).all():
                continue
            coerced = pd.to_numeric(series, errors="coerce")
            values = coerced.to_numpy(dtype=float)
            if np.isfinite(values).any():
                metric_cols.append(col)
    return metric_cols


def _coerce_metric_frame(frame: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    if not metric_cols:
        return frame
    numeric_cols = [
        col for col in metric_cols if not pd.api.types.is_numeric_dtype(frame[col])
    ]
    if not numeric_cols:
        return frame
    updated = frame.copy()
    updated[numeric_cols] = updated[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return updated


def export_results(
    results: MonteCarloResults,
    output_dir: Path | str,
    *,
    formats: Sequence[str] | str | None = None,
) -> dict[str, Path]:
    """Export results and summary frames to disk.

    Parameters
    ----------
    results:
        Aggregated Monte Carlo results.
    output_dir:
        Directory to write output files.
    formats:
        Iterable of formats (csv, json, parquet). Defaults to ("csv",).
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt_list = _coerce_formats(formats)
    exported: dict[str, Path] = {}
    for fmt in fmt_list:
        ext = fmt.lower()
        results_path = out_dir / f"results.{ext}"
        summary_path = out_dir / f"summary.{ext}"
        _export_frame(results.results_frame, results_path, ext)
        _export_frame(results.summary_frame, summary_path, ext)
        if results.cross_fold_summary_frame is not None:
            cross_path = out_dir / f"cross_fold_summary.{ext}"
            _export_frame(results.cross_fold_summary_frame, cross_path, ext)
            exported[f"cross_fold_summary_{ext}"] = cross_path
        if results.pooled_summary_frame is not None:
            pooled_path = out_dir / f"pooled_summary.{ext}"
            _export_frame(results.pooled_summary_frame, pooled_path, ext)
            exported[f"pooled_summary_{ext}"] = pooled_path
        if results.pooled_distribution_frame is not None:
            pooled_path = out_dir / f"pooled_distributions.{ext}"
            _export_frame(results.pooled_distribution_frame, pooled_path, ext)
            exported[f"pooled_distributions_{ext}"] = pooled_path
        exported[f"results_{ext}"] = results_path
        exported[f"summary_{ext}"] = summary_path
    return exported


def _coerce_formats(formats: Sequence[str] | str | None) -> list[str]:
    if formats is None:
        return ["csv"]
    if isinstance(formats, str):
        items = formats.split(",")
    else:
        items = list(formats)
    cleaned: list[str] = []
    for item in items:
        label = str(item).strip().lower()
        if not label:
            continue
        cleaned.append(label)
    return cleaned or ["csv"]


def _export_frame(frame: pd.DataFrame, path: Path, fmt: str) -> None:
    if fmt == "csv":
        frame.to_csv(path, index=False)
        return
    if fmt == "json":
        frame.to_json(path, orient="records")
        return
    if fmt == "parquet":
        frame.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported export format '{fmt}'")
