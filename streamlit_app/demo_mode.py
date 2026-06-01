"""Demo-mode helpers for the Streamlit app."""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "on"}


def demo_mode_enabled() -> bool:
    """Return True when the app is running in the synthetic-only demo zone."""

    return os.environ.get("TREND_DEMO_MODE", "").strip().lower() in _TRUTHY
