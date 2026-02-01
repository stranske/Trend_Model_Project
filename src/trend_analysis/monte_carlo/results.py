"""Result containers and export helpers for Monte Carlo simulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

__all__ = [
    "MonteCarloPathError",
    "MonteCarloResults",
    "StrategyEvaluation",
    "build_results_frame",
    "build_summary_frame",
    "export_results",
]


@dataclass(frozen=True)
class StrategyEvaluation:
    """Single strategy evaluation for one Monte Carlo path."""

    path_id: int
    strategy_name: str
    metrics: Mapping[str, float]
    metric_source: str | None
    path_hash: str
    seed: int | None = None
    diagnostic: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class MonteCarloPathError:
    """Error record for a failed path evaluation."""

    path_id: int
    strategy_name: str | None
    error_type: str
    message: str


@dataclass(frozen=True)
class MonteCarloResults:
    """Container for Monte Carlo evaluation outputs."""

    mode: str
    evaluations: Sequence[StrategyEvaluation]
    errors: Sequence[MonteCarloPathError]
    results_frame: pd.DataFrame
    summary_frame: pd.DataFrame
    metadata: Mapping[str, Any] | None = None


def build_results_frame(evaluations: Iterable[StrategyEvaluation]) -> pd.DataFrame:
    """Return a flat results table for all strategy evaluations."""

    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        row: dict[str, Any] = {
            "path_id": int(evaluation.path_id),
            "strategy": evaluation.strategy_name,
            "path_hash": evaluation.path_hash,
            "seed": evaluation.seed,
            "metric_source": evaluation.metric_source,
        }
        row.update({str(k): float(v) for k, v in evaluation.metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary_frame(results_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results per strategy."""

    if results_frame.empty:
        return pd.DataFrame()
    numeric_cols = results_frame.select_dtypes(include="number").columns.tolist()
    if "path_id" in numeric_cols:
        numeric_cols.remove("path_id")
    if "seed" in numeric_cols:
        numeric_cols.remove("seed")
    grouped = results_frame.groupby("strategy", dropna=False)
    summary = grouped[numeric_cols].mean(numeric_only=True)
    summary["paths"] = grouped.size()
    return summary.reset_index()


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
        exported[f"results_{ext}"] = results_path
        exported[f"summary_{ext}"] = summary_path
    return exported


def _coerce_formats(formats: Sequence[str] | str | None) -> list[str]:
    if formats is None:
        return ["csv"]
    if isinstance(formats, str):
        items = [formats]
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
