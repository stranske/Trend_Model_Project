"""Track configuration read/validation coverage for alignment checks."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Iterable

__all__ = [
    "ConfigCoverageReport",
    "ConfigCoverageTracker",
    "activate_config_coverage",
    "deactivate_config_coverage",
    "get_config_coverage_tracker",
    "wrap_config_for_coverage",
]


@dataclass(frozen=True, slots=True)
class ConfigCoverageReport:
    """Summary of config coverage activity."""

    read: set[str]
    validated: set[str]
    ignored: set[str]

    @property
    def unread_validated(self) -> set[str]:
        return self.validated - self.read

    @property
    def unvalidated_reads(self) -> set[str]:
        return self.read - self.validated


class ConfigCoverageTracker:
    """Record which config keys were read vs validated."""

    def __init__(self) -> None:
        self._read: set[str] = set()
        self._validated: set[str] = set()

    def track_read(self, key: str) -> None:
        if key:
            self._read.add(key)

    def track_validated(self, key: str) -> None:
        if key:
            self._validated.add(key)

    def generate_report(self) -> ConfigCoverageReport:
        read = set(self._read)
        validated = set(self._validated)
        ignored = (validated - read) | (read - validated)
        return ConfigCoverageReport(read=read, validated=validated, ignored=ignored)

    def format_report(self, report: ConfigCoverageReport | None = None) -> str:
        if report is None:
            report = self.generate_report()
        lines = ["Config coverage report:"]
        lines.append(f"  validated: {len(report.validated)}")
        lines.append(f"  read: {len(report.read)}")
        if report.unread_validated:
            lines.append("  validated-not-read:")
            lines.extend(f"    - {item}" for item in sorted(report.unread_validated))
        if report.unvalidated_reads:
            lines.append("  read-not-validated:")
            lines.extend(f"    - {item}" for item in sorted(report.unvalidated_reads))
        if not report.unread_validated and not report.unvalidated_reads:
            lines.append("  mismatches: none")
        return "\n".join(lines)


_ACTIVE_TRACKER: ConfigCoverageTracker | None = None


def activate_config_coverage(tracker: ConfigCoverageTracker) -> None:
    global _ACTIVE_TRACKER
    _ACTIVE_TRACKER = tracker


def deactivate_config_coverage() -> None:
    global _ACTIVE_TRACKER
    _ACTIVE_TRACKER = None


def get_config_coverage_tracker() -> ConfigCoverageTracker | None:
    return _ACTIVE_TRACKER


class _TrackedMapping(MutableMapping[str, Any]):
    """Mapping wrapper that records key access."""

    def __init__(
        self,
        data: Mapping[str, Any],
        prefix: str,
        tracker: ConfigCoverageTracker,
    ) -> None:
        self._data = data
        self._prefix = prefix
        self._tracker = tracker

    def _full_key(self, key: str) -> str:
        if self._prefix:
            return f"{self._prefix}.{key}"
        return key

    def _wrap_value(self, key: str, value: Any) -> Any:
        if isinstance(value, _TrackedMapping):
            return value
        if isinstance(value, Mapping):
            return _TrackedMapping(value, key, self._tracker)
        return value

    def __getitem__(self, key: str) -> Any:
        full_key = self._full_key(str(key))
        self._tracker.track_read(full_key)
        value = self._data[key]
        return self._wrap_value(full_key, value)

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(self._data, MutableMapping):
            self._data[key] = value
            return
        raise TypeError("Tracked mapping does not support item assignment")

    def __delitem__(self, key: str) -> None:
        if isinstance(self._data, MutableMapping):
            del self._data[key]
            return
        raise TypeError("Tracked mapping does not support item deletion")

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            self._tracker.track_read(self._full_key(key))
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        full_key = self._full_key(str(key))
        self._tracker.track_read(full_key)
        value = self._data.get(key, default)
        return self._wrap_value(full_key, value)

    def items(self) -> Iterable[tuple[str, Any]]:  # type: ignore[override]
        for key, value in self._data.items():
            full_key = self._full_key(str(key))
            self._tracker.track_read(full_key)
            yield key, self._wrap_value(full_key, value)

    def values(self) -> Iterable[Any]:  # type: ignore[override]
        for key, value in self._data.items():
            full_key = self._full_key(str(key))
            self._tracker.track_read(full_key)
            yield self._wrap_value(full_key, value)


def wrap_config_for_coverage(cfg: Any, tracker: ConfigCoverageTracker) -> Any:
    """Wrap config sections with read-tracking mappings."""

    field_names: Iterable[str]
    fields = getattr(cfg, "model_fields", None) or getattr(cfg, "__fields__", None)
    if fields:
        try:
            field_names = list(fields.keys())
        except AttributeError:
            field_names = list(fields)
    else:
        field_names = list(getattr(cfg, "__dict__", {}).keys())

    for name in field_names:
        try:
            value = getattr(cfg, name)
        except Exception:
            continue
        if isinstance(value, Mapping) and not isinstance(value, _TrackedMapping):
            wrapped = _TrackedMapping(value, name, tracker)
            try:
                setattr(cfg, name, wrapped)
            except Exception:
                pass
    return cfg
