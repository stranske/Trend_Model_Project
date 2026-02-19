"""Helpers for lightweight stage timing logs."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator

_PERF_LOGGER = logging.getLogger("trend_analysis.performance")


def log_timing(
    stage: str,
    *,
    duration_s: float,
    status: str | None = None,
    **fields: Any,
) -> None:
    """Emit a single-line timing entry for *stage*.

    Parameters
    ----------
    stage:
        Logical step name (e.g. ``"rolling_cache"``).
    duration_s:
        Elapsed duration in seconds.
    status:
        Optional status keyword such as ``"hit"`` or ``"miss"``.
    **fields:
        Additional key/value metadata appended to the log line.
    """

    if not _PERF_LOGGER.isEnabledFor(logging.INFO):  # fast path when disabled
        return

    parts = [f"stage={stage}", f"duration_ms={duration_s * 1000:.3f}"]
    if status:
        parts.append(f"status={status}")
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    _PERF_LOGGER.info(" ".join(parts))


@contextmanager
def timed_stage(
    stage: str,
    *,
    status: str | None = None,
    **fields: Any,
) -> Iterator[dict[str, Any]]:
    """Context manager that logs elapsed time upon exit.

    The yielded dictionary can be mutated by callers to update the
    ``status`` (for example changing a provisional "miss" to "hit").
    Additional metadata can also be appended via ``result["extra"]``.
    """

    state: dict[str, Any] = {"status": status}
    start = perf_counter()
    try:
        yield state
    finally:
        extra = fields.copy()
        status_value: str | None
        if (maybe_status := state.get("status")) is not None:
            status_value = str(maybe_status)
        else:
            status_value = status
        additional = state.get("extra")
        if isinstance(additional, dict):
            extra.update(additional)
        log_timing(
            stage,
            duration_s=perf_counter() - start,
            status=status_value,
            **extra,
        )


__all__ = ["log_timing", "timed_stage"]
