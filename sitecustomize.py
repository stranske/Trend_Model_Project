"""Opt-in shim for the Trend Model ``sitecustomize`` hooks."""

from __future__ import annotations

import os

_FLAG = "TREND_MODEL_SITE_CUSTOMIZE"


def _load_opt_in_module() -> None:
    """Import and execute the Trend Model bootstrap when enabled."""

    module = __import__("trend_model._sitecustomize", fromlist=["bootstrap"])
    bootstrap = getattr(module, "bootstrap")
    bootstrap()


if os.getenv(_FLAG) == "1":
    _load_opt_in_module()
