"""Deterministic diagnostics for explain-results prompts."""

from __future__ import annotations

import math
import numbers
import os
from collections.abc import Iterable, Mapping
from typing import Any

from .result_metrics import MetricEntry

_TURNOVER_WARN_ENV = "TREND_EXPLAIN_TURNOVER_WARN"
_MAX_WEIGHT_WARN_ENV = "TREND_EXPLAIN_MAX_WEIGHT_WARN"
_MAX_DD_WARN_ENV = "TREND_EXPLAIN_MAX_DD_WARN"

_DEFAULT_TURNOVER_WARN = 0.75
_DEFAULT_MAX_WEIGHT_WARN = 0.20
_DEFAULT_MAX_DD_WARN = -0.20

_TURNOVER_PATHS = (
    "risk_diagnostics.turnover_value",
    "risk_diagnostics.turnover",
    "turnover.mean",
    "turnover.latest",
    "turnover.value",
)
_WEIGHT_PREFIXES = ("fund_weights.", "ew_weights.")
_DRAWDOWN_PATHS = ("out_user_stats.max_drawdown", "out_ew_stats.max_drawdown")
_MISSING_SECTIONS = ("out_user_stats", "out_sample_scaled", "period_results")


def build_deterministic_feedback(
    details: Mapping[str, Any],
    entries: list[MetricEntry],
) -> str:
    """Build a deterministic diagnostics block for explain-results prompts."""

    bullets: list[str] = []
    entry_map = {entry.path: entry for entry in entries}

    turnover_warn = _read_env_float(_TURNOVER_WARN_ENV, default=_DEFAULT_TURNOVER_WARN)
    max_weight_warn = _read_env_float(_MAX_WEIGHT_WARN_ENV, default=_DEFAULT_MAX_WEIGHT_WARN)
    max_dd_warn = _read_env_float(_MAX_DD_WARN_ENV, default=_DEFAULT_MAX_DD_WARN)

    turnover_entry = _first_entry(entry_map, _TURNOVER_PATHS)
    turnover_value = _as_float(turnover_entry.value) if turnover_entry else None
    if turnover_entry and turnover_value is not None and turnover_value >= turnover_warn:
        source = turnover_entry.source or "unknown"
        bullets.append(
            "High turnover: "
            f"{_format_number(turnover_value)} exceeds {_format_number(turnover_warn)} "
            f"[from {source}]"
        )

    weight_entries = _select_weight_entries(entries)
    max_weight_entry, max_weight_value = _max_abs_weight(weight_entries)
    if max_weight_entry and max_weight_value is not None and max_weight_value >= max_weight_warn:
        fund_label = _fund_from_weight_path(max_weight_entry.path)
        source = max_weight_entry.source or "unknown"
        bullets.append(
            "High concentration: max abs weight "
            f"{_format_number(max_weight_value)}"
            f"{_format_optional_label(fund_label)} exceeds "
            f"{_format_number(max_weight_warn)} [from {source}]"
        )

    drawdown_entry = _first_entry(entry_map, _DRAWDOWN_PATHS)
    drawdown_value = _as_float(drawdown_entry.value) if drawdown_entry else None
    if drawdown_entry and drawdown_value is not None and drawdown_value <= max_dd_warn:
        source = drawdown_entry.source or "unknown"
        bullets.append(
            "Large drawdown: "
            f"{_format_number(drawdown_value)} is below {_format_number(max_dd_warn)} "
            f"[from {source}]"
        )

    missing_sections = _collect_missing_sections(details)
    if missing_sections:
        missing_count = len(missing_sections)
        bullets.append(
            "Missing result sections: "
            f"{', '.join(missing_sections)} (missing={missing_count}) [from details]"
        )

    if not bullets:
        return ""

    limited = bullets[:8]
    lines = ["Deterministic diagnostics:"]
    lines.extend(f"- {bullet}" for bullet in limited)
    return "\n".join(lines).strip()


def _read_env_float(name: str, *, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}.") from exc


def _first_entry(
    entry_map: Mapping[str, MetricEntry],
    paths: Iterable[str],
) -> MetricEntry | None:
    for path in paths:
        entry = entry_map.get(path)
        if entry is not None:
            return entry
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    return None


def _format_number(value: float) -> str:
    return str(value)


def _select_weight_entries(entries: Iterable[MetricEntry]) -> list[MetricEntry]:
    weights = [entry for entry in entries if entry.path.startswith(_WEIGHT_PREFIXES)]
    if any(entry.path.startswith("fund_weights.") for entry in weights):
        return [entry for entry in weights if entry.path.startswith("fund_weights.")]
    return weights


def _max_abs_weight(
    entries: Iterable[MetricEntry],
) -> tuple[MetricEntry | None, float | None]:
    max_entry: MetricEntry | None = None
    max_value: float | None = None
    for entry in entries:
        value = _as_float(entry.value)
        if value is None:
            continue
        abs_value = abs(value)
        if max_value is None or abs_value > max_value:
            max_value = abs_value
            max_entry = entry
    return max_entry, max_value


def _fund_from_weight_path(path: str) -> str:
    parts = path.split(".", maxsplit=1)
    return parts[1] if len(parts) > 1 else ""


def _format_optional_label(label: str) -> str:
    if not label:
        return ""
    return f" ({label})"


def _collect_missing_sections(details: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    multi_period_expected = "period_count" in details or "period_results" in details
    for section in _MISSING_SECTIONS:
        if section == "period_results" and not multi_period_expected:
            continue
        if _is_missing(details.get(section)):
            missing.append(section)
    return missing


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, Mapping):
        return len(value) == 0
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    if hasattr(value, "empty"):
        try:
            return bool(value.empty)
        except Exception:
            return False
    return False


__all__ = ["build_deterministic_feedback"]
