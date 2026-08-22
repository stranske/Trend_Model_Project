"""Test adapters over the supported diagnostics-aware pipeline entry point."""

from __future__ import annotations

from typing import Any

from trend_analysis.pipeline import run_analysis


def run_analysis_payload(*args: Any, **kwargs: Any) -> Any:
    """Return the payload while tests migrate away from the retired facade."""

    return run_analysis(*args, **kwargs).value
