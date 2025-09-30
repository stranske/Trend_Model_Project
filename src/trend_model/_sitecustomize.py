"""Optional bootstrap helpers mirroring the legacy repository
``sitecustomize``."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ENV_FLAG = "TREND_MODEL_SITE_CUSTOMIZE"
SITE_INDICATORS = {"site-packages", "dist-packages"}
PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_ROOT.parent
REPO_ROOT = SRC_DIR.parent

__all__: list[str] = ["ENV_FLAG", "SRC_DIR", "REPO_ROOT", "maybe_apply", "apply"]


def maybe_apply() -> None:
    """Apply the bootstrap helpers if the opt-in flag is enabled."""

    if os.getenv(ENV_FLAG) != "1":
        return
    apply()


def apply() -> None:
    """Inject ``src`` and validate our dependency resolution."""

    _ensure_src_on_sys_path()
    _ensure_joblib_external()


def _ensure_src_on_sys_path() -> None:
    if not SRC_DIR.exists():  # pragma: no cover - defensive for packaged installs
        return

    src = str(SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)


def _ensure_joblib_external() -> None:
    spec = importlib.util.find_spec("joblib")
    if spec is None or not spec.origin:
        # Dependency not installed yet (e.g. during bootstrapping); defer to the
        # eventual import which will raise a clearer ModuleNotFoundError.
        return

    resolved = Path(spec.origin).resolve()

    if any(part in SITE_INDICATORS for part in resolved.parts):
        # Virtual environments often live within the repository root (for
        # example ``.venv/``).  As long as the resolution path contains a
        # recognised site-packages/dist-packages segment we accept the module as
        # third-party.
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
