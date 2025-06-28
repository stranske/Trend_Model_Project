"""Execution engine for rolling multi-period runs (stub)."""

from __future__ import annotations
from typing import Any


from trend_analysis.config import Config  # pydantic model


def run(cfg: Config) -> dict[str, Any]:
    """Run the full multi-period back-test (Phase 2 implementation pending)."""
    raise NotImplementedError("Multi-period logic not implemented yet.")
