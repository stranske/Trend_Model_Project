"""Helpers for resolving risk-free configuration defaults consistently."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast


def resolve_risk_free_settings(
    data_settings: Mapping[str, Any] | None,
) -> tuple[str | None, bool]:
    """Determine risk-free column selection and fallback policy.

    The default behaviour disables the fallback heuristic unless the caller has
    explicitly opted into it by setting ``data.allow_risk_free_fallback`` to a
    boolean value. Supplying ``data.risk_free_column`` always takes precedence
    and disables the fallback regardless of the flag value to avoid silently
    overriding user intent.
    """

    if not data_settings:
        return None, False

    risk_free_column = cast(str | None, data_settings.get("risk_free_column"))
    allow_cfg = data_settings.get("allow_risk_free_fallback")

    if risk_free_column:
        return risk_free_column, False

    if isinstance(allow_cfg, bool):
        return risk_free_column, allow_cfg

    return risk_free_column, False


__all__ = ["resolve_risk_free_settings"]
