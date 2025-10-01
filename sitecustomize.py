"""Opt-in bootstrap shim for the Trend Model ``sitecustomize`` hooks."""

from __future__ import annotations

import os
from importlib import import_module
from types import ModuleType

ENV_FLAG = "TREND_MODEL_SITE_CUSTOMIZE"

__all__ = ["ENV_FLAG", "maybe_apply", "apply"]


def _load_opt_in_module() -> ModuleType:
    """Import the guarded bootstrap module lazily."""

    return import_module("trend_model._sitecustomize")


def maybe_apply() -> None:
    """Execute the optional bootstrap when the opt-in flag is present."""

    if os.getenv(ENV_FLAG) != "1":
        return

    apply()


def apply() -> None:
    """Proxy to the guarded bootstrap helpers without importing eagerly."""

    module = _load_opt_in_module()
    maybe_apply = getattr(module, "maybe_apply")
    maybe_apply()


# The ``sitecustomize`` shim is imported implicitly by Python when the
# repository root is on ``PYTHONPATH``.  To maintain the legacy behaviour for
# users who opt-in, we dispatch automatically when the flag is explicitly set.
maybe_apply()
