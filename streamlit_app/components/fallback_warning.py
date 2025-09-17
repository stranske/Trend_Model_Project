"""Helpers for displaying weight engine fallback warnings in Streamlit."""

from __future__ import annotations

from typing import Mapping, Any

__all__ = ["format_weight_engine_warning"]


def format_weight_engine_warning(
    fallback_info: Mapping[str, Any],
    *,
    suffix: str,
) -> str:
    """Return a consistent warning message for weight engine fallbacks."""
    engine = fallback_info.get("engine") or "unknown"
    error_type = fallback_info.get("error_type")
    error_msg = fallback_info.get("error")

    if error_type and error_msg:
        detail = f"{error_type}: {error_msg}"
    elif error_type:
        detail = str(error_type)
    elif error_msg:
        detail = str(error_msg)
    else:
        detail = "unknown error"

    return f"⚠️ Weight engine '{engine}' failed ({detail}). {suffix}".strip()
