"""
Rolling multi‑period driver – to be filled by the code‑gen agent.
"""

from __future__ import annotations
from typing import Dict, Mapping, Any

from .scheduler import generate_periods


def run(cfg: Mapping[str, Any]) -> Dict[str, object]:  # noqa: D401
    """Return generated periods and metadata for the given config."""

    periods = generate_periods(cfg)
    return {"periods": periods, "n_periods": len(periods)}
