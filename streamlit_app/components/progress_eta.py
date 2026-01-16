"""ETA helpers for progress bars."""

from __future__ import annotations

from typing import Tuple

DEFAULT_ETA_SECONDS = 25.0
MIN_ETA_SECONDS = 5.0
MAX_ETA_SECONDS = 90.0
DEFAULT_NEW_WEIGHT = 0.4
DEFAULT_MAX_RATIO = 0.95


def estimate_eta_seconds(
    stored: float | int | None,
    *,
    default: float = DEFAULT_ETA_SECONDS,
    min_seconds: float = MIN_ETA_SECONDS,
    max_seconds: float = MAX_ETA_SECONDS,
) -> float:
    """Clamp stored ETA values into a safe range or use the default."""
    if isinstance(stored, (int, float)) and stored > 0:
        return float(min(max(stored, min_seconds), max_seconds))
    return float(default)


def update_eta_seconds(
    stored: float | int | None,
    duration: float,
    *,
    new_weight: float = DEFAULT_NEW_WEIGHT,
) -> float | None:
    """Blend a new observation into the stored ETA value."""
    if duration <= 0:
        return None
    if isinstance(stored, (int, float)) and stored > 0:
        return float(stored * (1 - new_weight) + duration * new_weight)
    return float(duration)


def progress_ratio_and_remaining(
    elapsed: float,
    estimate: float,
    *,
    max_ratio: float = DEFAULT_MAX_RATIO,
) -> Tuple[float, float]:
    """Return progress ratio and remaining seconds for a running task."""
    if estimate <= 0:
        return 0.0, 0.0
    remaining = max(estimate - elapsed, 0.0)
    ratio = min(elapsed / estimate, max_ratio)
    return ratio, remaining
