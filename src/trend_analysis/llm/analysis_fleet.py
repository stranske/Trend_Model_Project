"""Fleet-record emission for deterministic Trend analysis runs."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import pandas as pd

from trend_analysis.llm.tracing import record_fleet_event, stable_hash


def _safe_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _config_fingerprint(config: Any) -> str:
    payload = {
        "sample_split": _safe_mapping(getattr(config, "sample_split", None)),
        "portfolio": _safe_mapping(getattr(config, "portfolio", None)),
        "metrics": _safe_mapping(getattr(config, "metrics", None)),
        "vol_adjust": _safe_mapping(getattr(config, "vol_adjust", None)),
        "run": _safe_mapping(getattr(config, "run", None)),
        "seed": getattr(config, "seed", None),
    }
    return stable_hash(payload)


def _dataset_id(frame: pd.DataFrame) -> str:
    date_bounds: dict[str, str | None] = {"start": None, "end": None}
    if "Date" in frame.columns:
        dates = pd.to_datetime(frame["Date"], errors="coerce").dropna()
        if not dates.empty:
            date_bounds = {
                "start": dates.min().date().isoformat(),
                "end": dates.max().date().isoformat(),
            }
    column_hashes = sorted(stable_hash(str(column)) for column in frame.columns)
    return stable_hash(
        {
            "rows": int(frame.shape[0]),
            "columns": len(column_hashes),
            "column_hashes": column_hashes,
            "date_bounds": date_bounds,
        }
    )


def _analysis_status(result: Any) -> str:
    diagnostic = getattr(result, "diagnostic", None)
    if diagnostic is not None:
        return str(getattr(diagnostic, "reason_code", "diagnostic"))
    metrics = getattr(result, "metrics", None)
    if isinstance(metrics, pd.DataFrame) and metrics.empty:
        return "empty"
    return "success"


def _artifact_ref(result: Any) -> str:
    metrics = getattr(result, "metrics", None)
    details = getattr(result, "details_sanitized", None)
    if details is None:
        details = getattr(result, "details", None)
    payload: dict[str, Any] = {"details_type": type(details).__name__}
    if isinstance(metrics, pd.DataFrame):
        payload["metrics"] = {
            "rows": int(metrics.shape[0]),
            "columns": [str(column) for column in metrics.columns],
            "index_hashes": sorted(stable_hash(str(index)) for index in metrics.index),
        }
    if isinstance(details, Mapping):
        payload["detail_keys"] = sorted(str(key) for key in details.keys())
    return stable_hash(payload)


def _safe_metric(result: Any) -> float | None:
    metrics = getattr(result, "metrics", None)
    if not isinstance(metrics, pd.DataFrame) or metrics.empty:
        return None
    for column in ("sharpe", "information_ratio", "cagr", "vol"):
        if column not in metrics.columns:
            continue
        values = pd.to_numeric(metrics[column], errors="coerce").dropna()
        if not values.empty:
            return round(float(values.mean()), 6)
    return None


def record_analysis_run(
    *,
    config: Any,
    returns: pd.DataFrame,
    result: Any,
    latency_ms: float | None = None,
) -> None:
    """Append a dashboard-safe fleet record for a deterministic analysis run."""

    if not isinstance(returns, pd.DataFrame):
        return

    status = "success" if os.environ.get("LANGSMITH_API_KEY") else "no_secret"
    analysis_status = _analysis_status(result)
    if analysis_status not in {"success", "empty"}:
        status = "error"

    domain = {
        "dataset_id": _dataset_id(returns),
        "config_fingerprint": _config_fingerprint(config),
        "analysis_status": analysis_status,
        "validation_status": "deterministic",
        "match_score": _safe_metric(result),
        "artifact_refs": {"analysis_summary": _artifact_ref(result)},
    }
    record_fleet_event(
        operation="analysis-run",
        status=status,
        provider="deterministic",
        model="trend-analysis",
        latency_ms=round(float(latency_ms), 3) if latency_ms is not None else None,
        domain=domain,
    )


__all__ = ["record_analysis_run"]
