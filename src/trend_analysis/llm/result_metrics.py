"""Extract metric catalogs from analysis results for NL explanations."""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

_BASE_STATS_FIELDS = (
    "cagr",
    "vol",
    "sharpe",
    "sortino",
    "information_ratio",
    "max_drawdown",
)
_OPTIONAL_STATS_FIELDS = ("is_avg_corr", "os_avg_corr")
_ALL_STATS_FIELDS = _BASE_STATS_FIELDS + _OPTIONAL_STATS_FIELDS

_STATS_SECTIONS = (
    "in_sample_stats",
    "out_sample_stats",
    "out_sample_stats_raw",
)
_SINGLE_STATS_SECTIONS = (
    "in_ew_stats",
    "out_ew_stats",
    "out_ew_stats_raw",
    "in_user_stats",
    "out_user_stats",
    "out_user_stats_raw",
)
_WEIGHT_SECTIONS = ("fund_weights", "ew_weights")
_BENCHMARK_SECTION = "benchmark_ir"
_RISK_DIAGNOSTICS_SECTION = "risk_diagnostics"
_RISK_DIAGNOSTICS_FIELDS = (
    "turnover",
    "turnover_value",
    "transaction_cost",
    "transaction_costs",
    "cost",
    "per_trade_bps",
    "half_spread_bps",
)
_TURNOVER_SUMMARY_SOURCE = "turnover_series"
_METRIC_SYNONYMS: dict[str, tuple[str, ...]] = {
    "cagr": ("cagr", "compound annual growth rate"),
    "vol": ("vol", "volatility"),
    "sharpe": ("sharpe", "sharpe ratio"),
    "sortino": ("sortino", "sortino ratio"),
    "information_ratio": ("information ratio", "information_ratio"),
    "max_drawdown": ("max drawdown", "drawdown", "max_drawdown"),
    "is_avg_corr": ("avg corr", "average correlation", "correlation"),
    "os_avg_corr": ("avg corr", "average correlation", "correlation"),
    "turnover": ("turnover", "turnover rate"),
    "turnover_value": ("turnover", "turnover value"),
    "transaction_cost": ("transaction cost", "transaction costs", "tx cost"),
    "transaction_costs": ("transaction cost", "transaction costs", "tx cost"),
    "cost": ("transaction cost", "transaction costs", "cost"),
    "per_trade_bps": ("transaction cost", "per trade bps", "per_trade_bps"),
    "half_spread_bps": ("transaction cost", "half spread bps", "half_spread_bps"),
    "weights": ("weights", "fund weights", "portfolio weights"),
    "benchmark_ir": ("benchmark ir", "benchmark information ratio"),
}
_KNOWN_METRIC_KEYWORDS = {
    "alpha",
    "beta",
    "tracking error",
    "turnover",
    "treynor",
    "calmar",
    "omega",
    "skew",
    "kurtosis",
    "transaction cost",
    "transaction costs",
}
_TOKEN_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class MetricEntry:
    path: str
    value: float | int | str
    source: str


def extract_metric_catalog(result: Mapping[str, Any]) -> list[MetricEntry]:
    """Extract scalar metrics from the analysis result payload."""

    entries: list[MetricEntry] = []

    for section in _STATS_SECTIONS:
        stats_map = result.get(section)
        if not isinstance(stats_map, Mapping):
            continue
        for name in _sorted_keys(stats_map):
            stats = stats_map[name]
            base = f"{section}.{name}"
            entries.extend(_extract_stats_entries(stats, base, section))

    for section in _SINGLE_STATS_SECTIONS:
        stats = result.get(section)
        if stats is None:
            continue
        entries.extend(_extract_stats_entries(stats, section, section))

    for section in _WEIGHT_SECTIONS:
        weights = result.get(section)
        if not isinstance(weights, Mapping):
            continue
        for name in _sorted_keys(weights):
            entry = _make_entry(f"{section}.{name}", weights[name], section)
            if entry is not None:
                entries.append(entry)

    benchmarks = result.get(_BENCHMARK_SECTION)
    if isinstance(benchmarks, Mapping):
        for bench in _sorted_keys(benchmarks):
            bench_map = benchmarks[bench]
            if not isinstance(bench_map, Mapping):
                continue
            for label in _sorted_keys(bench_map):
                entry = _make_entry(
                    f"{_BENCHMARK_SECTION}.{bench}.{label}",
                    bench_map[label],
                    _BENCHMARK_SECTION,
                )
                if entry is not None:
                    entries.append(entry)

    entries.extend(_extract_risk_diagnostics_entries(result))
    entries.extend(_extract_turnover_series_entries(result))

    return entries


def format_metric_catalog(entries: Iterable[MetricEntry]) -> str:
    """Render metric entries into a readable catalog string."""

    lines = [f"{entry.path}: {entry.value} [from {entry.source}]" for entry in entries]
    return "\n".join(lines).strip()


def available_metric_keywords(entries: Iterable[MetricEntry]) -> set[str]:
    """Return normalized metric keywords available in the metric catalog."""

    keywords: set[str] = set()
    for entry in entries:
        for label in _metric_labels_for_entry(entry):
            keywords.add(_normalize_metric_label(label))
    return {keyword for keyword in keywords if keyword}


def known_metric_keywords() -> set[str]:
    """Return a normalized set of known metric keywords to detect missing requests."""

    keywords = set(_KNOWN_METRIC_KEYWORDS)
    for aliases in _METRIC_SYNONYMS.values():
        keywords.update(aliases)
    return {_normalize_metric_label(keyword) for keyword in keywords if keyword}


def _extract_stats_entries(stats: Any, base: str, source: str) -> list[MetricEntry]:
    stats_map = _stats_to_mapping(stats)
    entries: list[MetricEntry] = []
    for field in _ALL_STATS_FIELDS:
        if field not in stats_map:
            continue
        entry = _make_entry(f"{base}.{field}", stats_map[field], source)
        if entry is not None:
            entries.append(entry)
    return entries


def _stats_to_mapping(stats: Any) -> dict[str, Any]:
    if isinstance(stats, Mapping):
        return {str(key): value for key, value in stats.items()}

    if isinstance(stats, tuple):
        if len(stats) >= len(_BASE_STATS_FIELDS):
            return {
                field: value
                for field, value in zip(_BASE_STATS_FIELDS, stats[: len(_BASE_STATS_FIELDS)])
            }
        return {}

    if hasattr(stats, "__dict__"):
        data = {key: getattr(stats, key) for key in _ALL_STATS_FIELDS if hasattr(stats, key)}
        return data

    return {}


def _make_entry(path: str, value: Any, source: str) -> MetricEntry | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        if isinstance(value, float) and math.isnan(value):
            return None
        return MetricEntry(path=path, value=float(value), source=source)
    if isinstance(value, str):
        return MetricEntry(path=path, value=value, source=source)
    return None


def _sorted_keys(mapping: Mapping[str, Any]) -> list[str]:
    return sorted((str(key) for key in mapping.keys()))


def _metric_labels_for_entry(entry: MetricEntry) -> Iterable[str]:
    path_parts = entry.path.split(".")
    metric_label = path_parts[-1] if path_parts else entry.path
    if entry.path.startswith("turnover."):
        metric_label = "turnover"
    if entry.source in _WEIGHT_SECTIONS:
        metric_label = "weights"
    elif entry.source == _BENCHMARK_SECTION:
        metric_label = "benchmark_ir"

    synonyms = _METRIC_SYNONYMS.get(metric_label, (metric_label,))
    return synonyms


def _normalize_metric_label(label: str) -> str:
    normalized = _TOKEN_RE.sub(" ", label.lower()).strip()
    return " ".join(normalized.split())


def _extract_risk_diagnostics_entries(result: Mapping[str, Any]) -> list[MetricEntry]:
    diagnostics = result.get(_RISK_DIAGNOSTICS_SECTION)
    diag_map = _diagnostics_to_mapping(diagnostics)
    entries: list[MetricEntry] = []
    for field in _RISK_DIAGNOSTICS_FIELDS:
        if field not in diag_map:
            continue
        entry = _make_entry(
            f"{_RISK_DIAGNOSTICS_SECTION}.{field}",
            diag_map[field],
            _RISK_DIAGNOSTICS_SECTION,
        )
        if entry is not None:
            entries.append(entry)
    return entries


def _diagnostics_to_mapping(diagnostics: Any) -> dict[str, Any]:
    if isinstance(diagnostics, Mapping):
        return {str(key): value for key, value in diagnostics.items()}
    if diagnostics is None:
        return {}
    return {
        field: getattr(diagnostics, field)
        for field in _RISK_DIAGNOSTICS_FIELDS
        if hasattr(diagnostics, field)
    }
    return {}


def _extract_turnover_series_entries(result: Mapping[str, Any]) -> list[MetricEntry]:
    turnover_obj = result.get("turnover")
    if turnover_obj is None:
        details = result.get("details")
        if isinstance(details, Mapping):
            turnover_obj = details.get("turnover")
    series = _coerce_series(turnover_obj)
    if series is None or series.empty:
        return []
    series = series.dropna()
    if series.empty:
        return []
    latest = float(series.iloc[-1])
    mean = float(series.mean())
    return [
        MetricEntry(path="turnover.latest", value=latest, source=_TURNOVER_SUMMARY_SOURCE),
        MetricEntry(path="turnover.mean", value=mean, source=_TURNOVER_SUMMARY_SOURCE),
    ]


def _coerce_series(value: Any) -> pd.Series | None:
    if isinstance(value, pd.Series):
        return value.astype(float).copy()
    if isinstance(value, Mapping):
        if not value:
            return None
        return pd.Series(value, dtype=float)
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return pd.Series(value, dtype=float)
    return None


__all__ = [
    "MetricEntry",
    "available_metric_keywords",
    "extract_metric_catalog",
    "format_metric_catalog",
    "known_metric_keywords",
]
