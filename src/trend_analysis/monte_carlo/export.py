"""Export helpers for Monte Carlo aggregation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .aggregator import (
    AGGREGATION_PATH_COLUMNS,
    BREACH_FRAME_SCHEMA,
    EXPECTED_SHORTFALL_FRAME_SCHEMA,
    QUANTILES_FRAME_SCHEMA,
    MonteCarloAggregationResults,
)

__all__ = ["export_aggregation_results"]


def export_aggregation_results(
    results: MonteCarloAggregationResults,
    output_dir: Path | str,
    *,
    formats: Sequence[str] | str | None = None,
) -> dict[str, Path]:
    """Export aggregation frames to disk.

    Parameters
    ----------
    results:
        Aggregated Monte Carlo distribution summaries.
    output_dir:
        Directory to write output files.
    formats:
        Iterable of formats (csv, parquet). Defaults to ("csv",).
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt_list = _coerce_formats(formats)
    exported: dict[str, Path] = {}
    path_frame = _reorder_path_frame(results.path_frame)
    quantiles_frame = _reorder_schema_frame(results.quantiles_frame, QUANTILES_FRAME_SCHEMA)
    breach_frame = _reorder_schema_frame(results.breach_frame, BREACH_FRAME_SCHEMA)
    shortfall_frame = _reorder_schema_frame(
        results.expected_shortfall_frame,
        EXPECTED_SHORTFALL_FRAME_SCHEMA,
    )
    for fmt in fmt_list:
        ext = fmt.lower()
        path_path = out_dir / f"path_summary.{ext}"
        quantiles_path = out_dir / f"quantiles.{ext}"
        breach_path = out_dir / f"breach_probabilities.{ext}"
        shortfall_path = out_dir / f"expected_shortfall.{ext}"

        _export_frame(path_frame, path_path, ext)
        _export_frame(quantiles_frame, quantiles_path, ext)
        _export_frame(breach_frame, breach_path, ext)
        _export_frame(shortfall_frame, shortfall_path, ext)

        exported[f"path_summary_{ext}"] = path_path
        exported[f"quantiles_{ext}"] = quantiles_path
        exported[f"breach_probabilities_{ext}"] = breach_path
        exported[f"expected_shortfall_{ext}"] = shortfall_path
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
    if not cleaned:
        return ["csv"]
    deduped: list[str] = []
    seen: set[str] = set()
    for label in cleaned:
        if label in seen:
            continue
        seen.add(label)
        deduped.append(label)
    return deduped


def _export_frame(frame: pd.DataFrame, path: Path, fmt: str) -> None:
    if fmt == "csv":
        frame.to_csv(path, index=False)
        return
    if fmt == "parquet":
        frame.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported export format '{fmt}'")


def _reorder_path_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [col for col in AGGREGATION_PATH_COLUMNS if col not in frame.columns]
    if missing_cols:
        frame = frame.copy()
        for col in missing_cols:
            frame[col] = pd.NA
    base_cols = list(AGGREGATION_PATH_COLUMNS)
    metric_cols = [col for col in frame.columns if col not in base_cols]
    return frame[base_cols + metric_cols]


def _reorder_schema_frame(frame: pd.DataFrame, schema: Sequence[str]) -> pd.DataFrame:
    missing_cols = [col for col in schema if col not in frame.columns]
    if missing_cols:
        frame = frame.copy()
        for col in missing_cols:
            frame[col] = pd.NA
    schema_cols = [col for col in schema if col in frame.columns]
    extra_cols = [col for col in frame.columns if col not in schema_cols]
    return frame[schema_cols + extra_cols]
