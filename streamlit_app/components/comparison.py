"""Comparison helpers for A/B result evaluation in the Streamlit app."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from io import BytesIO
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd


def comparison_run_key(
    model_state: dict[str, Any],
    *,
    benchmark: str | None,
    funds: Iterable[str],
    data_fingerprint: str | None,
    info_ratio_benchmark: str | None = None,
    risk_free: str | None = None,
) -> str:
    """Build a stable cache key for comparison runs."""

    payload = {
        "model": _hashable_model_state(model_state),
        "benchmark": benchmark or "__none__",
        "funds": sorted(str(f) for f in funds),
        "data": data_fingerprint or "unknown",
        "info_ratio_benchmark": info_ratio_benchmark or "__none__",
        "risk_free": risk_free or "__none__",
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _hashable_model_state(state: Mapping[str, Any]) -> str:
    return json.dumps(state, sort_keys=True, default=str)


def _coerce_numeric(value: Any) -> float | None:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return as_float if np.isfinite(as_float) else None


def _numeric_series_from_metrics(result) -> pd.Series:
    metrics = getattr(result, "metrics", None)
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        series = metrics.iloc[0]
        numeric = {str(k): _coerce_numeric(v) for k, v in series.items()}
        return pd.Series({k: v for k, v in numeric.items() if v is not None})

    details = getattr(result, "details", {}) or {}
    stats_obj = details.get("out_sample_stats") or {}
    if hasattr(stats_obj, "items"):
        stats_map = dict(stats_obj.items())
    elif hasattr(stats_obj, "__dict__"):
        stats_map = dict(vars(stats_obj))
    else:
        stats_map = {}
    numeric = {str(k): _coerce_numeric(v) for k, v in stats_map.items()}
    return pd.Series({k: v for k, v in numeric.items() if v is not None})


def metric_delta_frame(
    result_a,
    result_b,
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> pd.DataFrame:
    """Build a side-by-side metric table with deltas."""

    series_a = _numeric_series_from_metrics(result_a)
    series_b = _numeric_series_from_metrics(result_b)
    all_metrics = sorted(set(series_a.index) | set(series_b.index))
    rows = []
    for name in all_metrics:
        a_val = series_a.get(name)
        b_val = series_b.get(name)
        delta = None
        if a_val is not None and b_val is not None:
            delta = b_val - a_val
        rows.append(
            {
                "Metric": name,
                label_a: a_val,
                label_b: b_val,
                "Delta (B - A)": delta,
            }
        )
    return pd.DataFrame(rows)


def _period_label(period: Iterable[Any], fallback: int) -> str:
    try:
        period_list = list(period)
    except Exception:
        return f"Period {fallback}"
    out_start = period_list[2] if len(period_list) > 2 else ""
    out_end = period_list[3] if len(period_list) > 3 else ""
    if out_start or out_end:
        return f"{out_start} → {out_end}".strip()
    return f"Period {fallback}"


def period_summary(result) -> pd.DataFrame:
    """Summarize multi-period outputs for comparison."""

    details = getattr(result, "details", {}) or {}
    period_results = details.get("period_results", []) or []
    rows = []
    for idx, res in enumerate(period_results):
        period = res.get("period", ())
        selected = res.get("selected_funds") or []
        turnover = _coerce_numeric(
            res.get("turnover") or (res.get("risk_diagnostics") or {}).get("turnover")
        )
        txn_cost = _coerce_numeric(res.get("transaction_cost"))
        rows.append(
            {
                "Period": _period_label(period, idx + 1),
                "Selected Funds": len(selected),
                "Turnover": turnover,
                "Transaction Cost": txn_cost,
            }
        )
    return pd.DataFrame(rows)


def _delta_frame(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    key: str,
    numeric_cols: Iterable[str],
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    merged = frame_a.set_index(key).join(
        frame_b.set_index(key), how="outer", lsuffix="_a", rsuffix="_b"
    )
    merged = merged.reset_index().rename(columns={"index": key})
    rows = []
    for _, row in merged.iterrows():
        payload = {key: row[key]}
        for col in numeric_cols:
            a_val = row.get(f"{col}_a")
            b_val = row.get(f"{col}_b")
            if pd.isna(a_val):
                a_val = None
            if pd.isna(b_val):
                b_val = None
            payload[f"{label_a} {col}"] = a_val
            payload[f"{label_b} {col}"] = b_val
            delta = None
            if a_val is not None and b_val is not None:
                delta = b_val - a_val
            payload[f"{col} Δ (B - A)"] = delta
        rows.append(payload)
    return pd.DataFrame(rows)


def period_delta(
    result_a,
    result_b,
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> pd.DataFrame:
    """Compare two period summaries with deltas."""

    summary_a = period_summary(result_a)
    summary_b = period_summary(result_b)
    if summary_a.empty and summary_b.empty:
        return pd.DataFrame()
    cols = ["Selected Funds", "Turnover", "Transaction Cost"]
    return _delta_frame(
        summary_a,
        summary_b,
        key="Period",
        numeric_cols=cols,
        label_a=label_a,
        label_b=label_b,
    )


def manager_change_counts(result) -> pd.DataFrame:
    """Count manager change reasons per result."""

    details = getattr(result, "details", {}) or {}
    period_results = details.get("period_results", []) or []
    counts: dict[str, int] = {}
    for res in period_results:
        changes = res.get("manager_changes") or []
        for change in changes:
            if not isinstance(change, dict):
                continue
            reason = str(change.get("reason") or "unspecified")
            counts[reason] = counts.get(reason, 0) + 1
    rows = [{"Reason": r, "Count": c} for r, c in sorted(counts.items())]
    return pd.DataFrame(rows)


def manager_change_delta(
    result_a,
    result_b,
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> pd.DataFrame:
    """Compare manager change counts with deltas."""

    counts_a = manager_change_counts(result_a)
    counts_b = manager_change_counts(result_b)
    if counts_a.empty and counts_b.empty:
        return pd.DataFrame()
    cols = ["Count"]
    return _delta_frame(
        counts_a,
        counts_b,
        key="Reason",
        numeric_cols=cols,
        label_a=label_a,
        label_b=label_b,
    )


def build_comparison_bundle(
    *,
    config_a: dict[str, Any],
    config_b: dict[str, Any],
    diff_text: str,
    metrics: pd.DataFrame | None = None,
    periods: pd.DataFrame | None = None,
    manager_changes: pd.DataFrame | None = None,
) -> bytes:
    """Create a ZIP bundle containing configs, diff text, and comparison CSVs."""

    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("config_A.json", json.dumps(config_a, indent=2, sort_keys=True, default=str))
        zf.writestr("config_B.json", json.dumps(config_b, indent=2, sort_keys=True, default=str))
        zf.writestr("config_diff.txt", diff_text or "No differences found.")

        if metrics is not None and not metrics.empty:
            zf.writestr("metrics_compare.csv", metrics.to_csv(index=False))
        if periods is not None and not periods.empty:
            zf.writestr("periods_compare.csv", periods.to_csv(index=False))
        if manager_changes is not None and not manager_changes.empty:
            zf.writestr("manager_change_compare.csv", manager_changes.to_csv(index=False))

    buffer.seek(0)
    return buffer.getvalue()
