"""Lightweight diagnostic payloads for early-exit signalling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Mapping,
    Protocol,
    TypeGuard,
    TypeVar,
    runtime_checkable,
)

try:  # pragma: no cover - optional instrumentation
    from trend_analysis.config.coverage import (
        ConfigCoverageTracker,
        get_config_coverage_tracker,
    )
except Exception:  # pragma: no cover - defensive fallback

    def get_config_coverage_tracker() -> ConfigCoverageTracker | None:
        return None

    ConfigCoverageTracker = None  # type: ignore[assignment,misc]


T = TypeVar("T")


@runtime_checkable
class RunPayload(Protocol[T]):
    """Canonical run payload contract for pipeline entry points."""

    value: T | None
    diagnostic: "DiagnosticPayload | None"
    metadata: Mapping[str, object] | None


@dataclass(slots=True)
class RunPayloadResult(Generic[T]):
    """Lightweight RunPayload carrier for simple use cases."""

    value: T | None
    diagnostic: "DiagnosticPayload | None" = None
    metadata: Mapping[str, object] | None = None


@dataclass(slots=True)
class DiagnosticPayload:
    """Structured context for early exits and skipped work."""

    reason_code: str
    message: str
    context: Mapping[str, object] | None = None


def is_run_payload(obj: object) -> TypeGuard[RunPayload[Any]]:
    """Return True when ``obj`` matches the RunPayload contract."""

    try:
        diagnostic = getattr(obj, "diagnostic")
        metadata = getattr(obj, "metadata")
        getattr(obj, "value")
    except Exception:
        return False

    if diagnostic is not None and not isinstance(diagnostic, DiagnosticPayload):
        return False
    if metadata is not None and not isinstance(metadata, Mapping):
        return False
    return True


@dataclass(slots=True)
class DiagnosticResult(Generic[T]):
    """Container pairing a value with an optional diagnostic."""

    value: T | None
    diagnostic: DiagnosticPayload | None = None
    coverage_report: object | None = None

    def __post_init__(self) -> None:
        if self.coverage_report is not None:
            return
        if not logging.getLogger().isEnabledFor(logging.DEBUG):
            return
        tracker = get_config_coverage_tracker()
        if tracker is None:
            return
        try:
            self.coverage_report = tracker.generate_report()
        except Exception:
            return

    @classmethod
    def success(cls, value: T) -> "DiagnosticResult[T]":
        return cls(value=value, diagnostic=None)

    @classmethod
    def failure(
        cls,
        *,
        reason_code: str,
        message: str,
        context: Mapping[str, object] | None = None,
    ) -> "DiagnosticResult[T]":
        return cls(value=None, diagnostic=DiagnosticPayload(reason_code, message, context))
