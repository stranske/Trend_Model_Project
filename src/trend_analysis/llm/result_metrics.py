"""Extract metric catalogs from analysis results for NL explanations."""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import re
from typing import Any

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
_METRIC_SYNONYMS: dict[str, tuple[str, ...]] = {
    "cagr": ("cagr", "compound annual growth rate"),
    "vol": ("vol", "volatility"),
    "sharpe": ("sharpe", "sharpe ratio"),
    "sortino": ("sortino", "sortino ratio"),
    "information_ratio": ("information ratio", "information_ratio"),
    "max_drawdown": ("max drawdown", "drawdown", "max_drawdown"),
    "is_avg_corr": ("avg corr", "average correlation", "correlation"),
    "os_avg_corr": ("avg corr", "average correlation", "correlation"),
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
    if entry.source in _WEIGHT_SECTIONS:
        metric_label = "weights"
    elif entry.source == _BENCHMARK_SECTION:
        metric_label = "benchmark_ir"

    synonyms = _METRIC_SYNONYMS.get(metric_label, (metric_label,))
    return synonyms


def _normalize_metric_label(label: str) -> str:
    normalized = _TOKEN_RE.sub(" ", label.lower()).strip()
    return " ".join(normalized.split())


__all__ = [
    "MetricEntry",
    "available_metric_keywords",
    "extract_metric_catalog",
    "format_metric_catalog",
    "known_metric_keywords",
]
