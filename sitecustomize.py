"""Opt-in bootstrap shim for the Trend Model ``sitecustomize`` hooks."""

from __future__ import annotations

import os

_FLAG = "TREND_MODEL_SITE_CUSTOMIZE"


def _load_opt_in_module() -> None:
    """Import and execute the Trend Model bootstrap when enabled."""

    module = __import__("trend_model._sitecustomize", fromlist=["maybe_apply"])
    maybe_apply = getattr(module, "maybe_apply")
    maybe_apply()


def _maybe_apply() -> None:
    """Execute the optional bootstrap when the flag is enabled."""

    if os.getenv(_FLAG) != "1":
        return

    _load_opt_in_module()


_maybe_apply()

__all__: list[str] = []
