"""Lightweight diagnostic payloads for early-exit signalling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Mapping, TypeVar

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
