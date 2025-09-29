"""Repository bootstrap hooks executed automatically during interpreter start.

The helper keeps ``src/`` on ``sys.path`` for the test-suite subprocesses and
verifies that ``joblib`` resolves to the third-party dependency rather than a
repository-local module.  Python automatically imports :mod:`sitecustomize` if
present on the import path during interpreter start.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SITE_INDICATORS = {"site-packages", "dist-packages"}
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"


def _ensure_src_on_sys_path() -> None:
    """Prepend ``src`` to ``sys.path`` if the directory exists."""

    if SRC_DIR.exists():  # pragma: no cover - trivial branch
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
    if REPO_ROOT in resolved.parents or resolved == REPO_ROOT:
        raise ImportError(
            "The third-party 'joblib' package is required; found repository "
            f"stub at {resolved}."
        )

    if not any(part in SITE_INDICATORS for part in resolved.parts):
        raise ImportError(
            "joblib should resolve from site-packages/dist-packages but instead "
            f"resolved to {resolved}."
        )


_ensure_src_on_sys_path()
_ensure_joblib_external()
