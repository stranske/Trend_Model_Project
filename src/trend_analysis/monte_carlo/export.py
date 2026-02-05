"""Export helpers for Monte Carlo aggregation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .aggregator import (
    AGGREGATION_PATH_COLUMNS,
    BREACH_FRAME_SCHEMA,
    EXPECTED_SHORTFALL_FRAME_SCHEMA,
    QUANTILES_FRAME_SCHEMA,
    MonteCarloAggregationResults,
)
from .results import RESULT_BASE_COLUMNS

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
        Iterable of formats (csv, parquet). Defaults to ("csv", "parquet") when
        parquet support is available, otherwise ("csv",).
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt_list = _coerce_formats(formats)
    exported: dict[str, Path] = {}
    path_frame = _reorder_path_frame(results.path_frame)
    quantiles_frame = _reorder_schema_frame(results.quantiles_frame, QUANTILES_FRAME_SCHEMA)
    breach_frame = _reorder_schema_frame(results.breach_frame, BREACH_FRAME_SCHEMA)
    summary_quantiles_frame = _build_summary_quantiles_frame(quantiles_frame)
    shortfall_frame = _reorder_schema_frame(
        results.expected_shortfall_frame,
        EXPECTED_SHORTFALL_FRAME_SCHEMA,
    )
    for fmt in fmt_list:
        ext = fmt.lower()
        path_path = out_dir / f"path_summary.{ext}"
        per_strategy_path = out_dir / f"per_strategy_stats.{ext}"
        per_strategy_path_alias = out_dir / f"per_strategy_path.{ext}"
        quantiles_path = out_dir / f"quantiles.{ext}"
        summary_quantiles_path = out_dir / f"summary_quantiles.{ext}"
        breach_path = out_dir / f"breach_probabilities.{ext}"
        shortfall_path = out_dir / f"expected_shortfall.{ext}"

        _export_frame(path_frame, path_path, ext)
        _export_frame(path_frame, per_strategy_path, ext)
        _export_frame(path_frame, per_strategy_path_alias, ext)
        _export_frame(quantiles_frame, quantiles_path, ext)
        _export_frame(summary_quantiles_frame, summary_quantiles_path, ext)
        _export_frame(breach_frame, breach_path, ext)
        _export_frame(shortfall_frame, shortfall_path, ext)

        exported[f"path_summary_{ext}"] = path_path
        exported[f"per_strategy_stats_{ext}"] = per_strategy_path
        exported[f"per_strategy_path_{ext}"] = per_strategy_path_alias
        exported[f"quantiles_{ext}"] = quantiles_path
        exported[f"summary_quantiles_{ext}"] = summary_quantiles_path
        exported[f"breach_probabilities_{ext}"] = breach_path
        exported[f"expected_shortfall_{ext}"] = shortfall_path
    return exported


def _coerce_formats(formats: Sequence[str] | str | None) -> list[str]:
    if formats is None:
        return _default_formats()
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
    if not cleaned:
        return _default_formats()
    deduped: list[str] = []
    seen: set[str] = set()
    for label in cleaned:
        if label in seen:
            continue
        seen.add(label)
        deduped.append(label)
    return _filter_unsupported_formats(deduped)


def _default_formats() -> list[str]:
    if _supports_parquet():
        return ["csv", "parquet"]
    return ["csv"]


def _filter_unsupported_formats(formats: list[str]) -> list[str]:
    if "parquet" not in formats:
        return formats
    if _supports_parquet():
        return formats
    supported = [fmt for fmt in formats if fmt != "parquet"]
    if supported:
        return supported
    raise ValueError("Parquet export requested but no parquet engine is available.")


def _supports_parquet() -> bool:
    return any(_module_available(module) for module in ("pyarrow", "fastparquet"))


def _module_available(module_name: str) -> bool:
    try:
        __import__(module_name)
    except Exception:
        return False
    return True


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
    if "strategy_name" in frame.columns:
        frame = frame.copy()
        if "strategy" not in frame.columns:
            frame["strategy"] = frame["strategy_name"]
        else:
            strategy = frame["strategy"]
            strategy_name = frame["strategy_name"]
            if _is_numeric_like(strategy) and not _is_numeric_like(strategy_name):
                frame["strategy"] = strategy_name
            else:
                frame["strategy"] = strategy.where(strategy.notna(), strategy_name)
    if "path" in frame.columns:
        if "path_id" in frame.columns:
            frame = frame.copy()
            frame["path"] = frame["path"].where(frame["path"].notna(), frame["path_id"])
        if "path_hash" in frame.columns:
            frame = frame.copy()
            frame["path"] = frame["path"].where(frame["path"].notna(), frame["path_hash"])
    elif "path_id" in frame.columns:
        frame = frame.copy()
        frame["path"] = frame["path_id"]
        if "path_hash" in frame.columns:
            frame["path"] = frame["path"].where(frame["path"].notna(), frame["path_hash"])
    elif "path_hash" in frame.columns:
        frame = frame.copy()
        frame["path"] = frame["path_hash"]
    if "fold" in frame.columns and "fold_id" in frame.columns:
        frame = frame.copy()
        frame["fold"] = frame["fold"].where(frame["fold"].notna(), frame["fold_id"])
    elif "fold" in frame.columns and "fold_id" not in frame.columns:
        pass
    elif "fold_id" in frame.columns:
        frame = frame.copy()
        frame["fold"] = frame["fold_id"]
    if "fold" in frame.columns and "fold_label" in frame.columns:
        frame = frame.copy()
        frame["fold"] = frame["fold"].where(frame["fold"].notna(), frame["fold_label"])
    elif "fold" not in frame.columns and "fold_label" in frame.columns:
        frame = frame.copy()
        frame["fold"] = frame["fold_label"]
    base_cols = list(AGGREGATION_PATH_COLUMNS)
    excluded = set(RESULT_BASE_COLUMNS)
    excluded.update({"path_id", "fold_id", "strategy_name", "paths", "folds"})
    metric_cols = [col for col in frame.columns if col not in base_cols and col not in excluded]
    return frame[base_cols + metric_cols]


def _is_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        non_null = series.dropna()
        if (
            not non_null.empty
            and non_null.map(lambda value: isinstance(value, (bool, np.bool_))).all()
        ):
            return False
        coerced = pd.to_numeric(series, errors="coerce")
        return bool(np.isfinite(coerced.to_numpy(dtype=float)).any())
    return False


def _reorder_schema_frame(frame: pd.DataFrame, schema: Sequence[str]) -> pd.DataFrame:
    missing_cols = [col for col in schema if col not in frame.columns]
    if missing_cols:
        frame = frame.copy()
        for col in missing_cols:
            frame[col] = pd.NA
    schema_cols = [col for col in schema if col in frame.columns]
    extra_cols = [col for col in frame.columns if col not in schema_cols]
    return frame[schema_cols + extra_cols]


def _build_summary_quantiles_frame(quantiles_frame: pd.DataFrame) -> pd.DataFrame:
    fold_col = None
    has_fold_id = "fold_id" in quantiles_frame.columns
    has_fold = "fold" in quantiles_frame.columns
    fold_id_has_values = has_fold_id and (
        quantiles_frame.empty or quantiles_frame["fold_id"].notna().any()
    )
    fold_has_values = has_fold and (quantiles_frame.empty or quantiles_frame["fold"].notna().any())
    if fold_id_has_values:
        fold_col = "fold_id"
    elif fold_has_values:
        fold_col = "fold"
    elif has_fold_id:
        fold_col = "fold_id"
    elif has_fold:
        fold_col = "fold"

    if quantiles_frame.empty:
        base_cols = ["strategy", "metric"]
        if fold_col:
            base_cols.insert(1, fold_col)
        return pd.DataFrame(columns=base_cols)

    frame = quantiles_frame.copy()
    if "strategy" not in frame.columns:
        frame["strategy"] = pd.NA
    if "metric" not in frame.columns:
        frame["metric"] = pd.NA
    if "quantile" not in frame.columns:
        frame["quantile"] = pd.NA
    if "value" not in frame.columns:
        frame["value"] = pd.NA

    frame["quantile"] = pd.to_numeric(frame["quantile"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["quantile"])
    if frame.empty:
        base_cols = ["strategy", "metric"]
        if fold_col:
            base_cols.insert(1, fold_col)
        return pd.DataFrame(columns=base_cols)

    frame["quantile_label"] = frame["quantile"].apply(_format_quantile_label)
    id_cols = ["strategy"]
    if fold_col and fold_col in frame.columns:
        id_cols.append(fold_col)
    id_cols.append("metric")

    quantile_order = frame[["quantile", "quantile_label"]].drop_duplicates().sort_values("quantile")
    quantile_cols = quantile_order["quantile_label"].tolist()
    summary = frame.pivot_table(
        index=id_cols,
        columns="quantile_label",
        values="value",
        aggfunc="first",
        dropna=False,
    ).reset_index()
    return summary[id_cols + quantile_cols]


def _format_quantile_label(quantile: float) -> str:
    percent = quantile * 100.0
    if not np.isfinite(percent):
        return "qnan"
    rounded = round(percent)
    if np.isclose(percent, rounded, atol=1e-8):
        value = int(rounded)
        if value < 10:
            return f"q{value:02d}"
        return f"q{value}"
    text = f"{percent:.6f}".rstrip("0").rstrip(".")
    text = text.replace(".", "_")
    return f"q{text}"
