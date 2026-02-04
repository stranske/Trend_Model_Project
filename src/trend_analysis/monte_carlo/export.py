"""Export helpers for Monte Carlo aggregation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .aggregator import MonteCarloAggregationResults

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
    for fmt in fmt_list:
        ext = fmt.lower()
        path_path = out_dir / f"path_summary.{ext}"
        quantiles_path = out_dir / f"quantiles.{ext}"
        breach_path = out_dir / f"breach_probabilities.{ext}"
        shortfall_path = out_dir / f"expected_shortfall.{ext}"

        _export_frame(results.path_frame, path_path, ext)
        _export_frame(results.quantiles_frame, quantiles_path, ext)
        _export_frame(results.breach_frame, breach_path, ext)
        _export_frame(results.expected_shortfall_frame, shortfall_path, ext)

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
    return cleaned or ["csv"]


def _export_frame(frame: pd.DataFrame, path: Path, fmt: str) -> None:
    if fmt == "csv":
        frame.to_csv(path, index=False)
        return
    if fmt == "parquet":
        frame.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported export format '{fmt}'")
