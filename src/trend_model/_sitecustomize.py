"""Optional interpreter bootstrap hooks for Trend Model development.

The helpers in this module replicate the legacy ``sitecustomize`` behaviour
while allowing callers to opt-in explicitly by setting the
``TREND_MODEL_SITE_CUSTOMIZE`` environment variable to ``"1"`` prior to
interpreter start.  Importing the module has no side effects; consumers should
call :func:`bootstrap` to execute the hooks.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

__all__ = ["bootstrap"]

SITE_INDICATORS = {"site-packages", "dist-packages"}
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
SRC_DIR = REPO_ROOT / "src"


def _ensure_src_on_sys_path() -> None:
    """Prepend ``src`` to ``sys.path`` if the directory exists."""

    if SRC_DIR.exists():  # pragma: no branch - trivial guard
        src = str(SRC_DIR)
        if src not in sys.path:
            sys.path.insert(0, src)


def _ensure_joblib_external() -> None:
    """Fail fast if :mod:`joblib` resolves to a repository-local module."""

    spec = importlib.util.find_spec("joblib")
    if spec is None or not spec.origin:
        # Dependency not installed yet (e.g. during bootstrapping); defer to the
        # actual import which will raise a clearer ModuleNotFoundError.
        return

    resolved = Path(spec.origin).resolve()
    resolved_parts = resolved.parts

    if any(part in SITE_INDICATORS for part in resolved_parts):
        # Virtual environments often live inside the repository root (e.g.
        # ``.venv/``). As long as the resolution path contains a recognised
        # site-packages/dist-packages segment we accept it as the third-party
        # dependency.
        return

    if REPO_ROOT in resolved.parents or resolved == REPO_ROOT:
        raise ImportError(
            "The third-party 'joblib' package is required; found repository "
            f"stub at {resolved}."
        )

    raise ImportError(
        "joblib should resolve from site-packages/dist-packages but instead "
        f"resolved to {resolved}."
    )


def bootstrap() -> None:
    """Execute the optional interpreter bootstrap hooks."""

    _ensure_src_on_sys_path()
    _ensure_joblib_external()
