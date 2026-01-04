"""Lightweight diagnostic payloads for early-exit signalling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generic, Mapping, TypeVar

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


@dataclass(slots=True)
class DiagnosticPayload:
    """Structured context for early exits and skipped work."""

    reason_code: str
    message: str
    context: Mapping[str, object] | None = None


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
        return cls(
            value=None, diagnostic=DiagnosticPayload(reason_code, message, context)
        )
