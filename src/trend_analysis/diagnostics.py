"""Shared diagnostics helpers for early-exit conditions.

The analysis pipeline historically returned ``None`` for early exits, leaving
callers without context.  This module centralises a lightweight payload format
that captures the reason and minimal context for those short-circuit paths so
downstream callers (CLI/API) can surface actionable messaging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class EarlyExit:
    """Diagnostic payload describing why processing stopped early."""

    code: str
    message: str
    stage: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "early_exit",
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
        }
        if self.details:
            payload["details"] = self.details
        return payload


def early_exit_payload(
    code: str,
    message: str,
    *,
    stage: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a serialisable diagnostic payload for an early-exit path."""

    return EarlyExit(code=code, message=message, stage=stage, details=dict(details or {})).to_payload()


def is_early_exit(payload: Any) -> bool:
    """Best-effort detection for early-exit diagnostics."""

    return bool(
        isinstance(payload, Mapping)
        and payload.get("status") == "early_exit"
        and payload.get("code")
        and payload.get("message")
    )


def normalise_early_exit(payload: Mapping[str, Any] | EarlyExit) -> dict[str, Any]:
    """Coerce an early-exit structure into the canonical payload shape."""

    if isinstance(payload, EarlyExit):
        return payload.to_payload()
    if not isinstance(payload, Mapping):
        return {"status": "early_exit", "code": "unknown", "message": "Unrecognised diagnostic payload", "stage": "unknown"}
    code = str(payload.get("code", "unknown"))
    message = str(payload.get("message", "")) or "Unknown early-exit reason"
    stage = str(payload.get("stage", "unknown"))
    details_obj = payload.get("details")
    details: dict[str, Any] = dict(details_obj) if isinstance(details_obj, Mapping) else {}
    return {
        "status": "early_exit",
        "code": code,
        "message": message,
        "stage": stage,
        "details": details,
    }
