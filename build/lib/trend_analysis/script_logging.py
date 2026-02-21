"""Utilities for initialising perf logging in standalone scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, TypeVar

from .logging_setup import setup_logging

T = TypeVar("T")
_DISABLE_ENV = "TREND_DISABLE_PERF_LOGS"


def _derive_app_name(module_file: str | None) -> str:
    if module_file:
        return Path(module_file).stem.replace("_", "-") or "script"
    argv0 = Path(sys.argv[0]) if sys.argv else Path("script")
    stem = argv0.stem or argv0.name or "script"
    return stem.replace("_", "-")


def setup_script_logging(
    *,
    app_name: str | None = None,
    module_file: str | None = None,
    announce: bool = True,
) -> Path | None:
    """Initialise the perf logger for ad-hoc scripts."""

    disable = os.environ.get(_DISABLE_ENV, "").strip().lower()
    if disable in {"1", "true", "yes"}:
        return None
    resolved_app = app_name or _derive_app_name(module_file)
    log_path = setup_logging(app_name=resolved_app)
    if announce:
        print(f"Run log ({resolved_app}): {log_path}")
    return log_path


def run_with_script_logging(
    func: Callable[..., T],
    *args: Any,
    app_name: str | None = None,
    module_file: str | None = None,
    announce: bool = True,
    **kwargs: Any,
) -> T:
    """Call *func* after initialising script logging."""

    setup_script_logging(
        app_name=app_name,
        module_file=module_file,
        announce=announce,
    )
    return func(*args, **kwargs)


__all__ = ["setup_script_logging", "run_with_script_logging"]
