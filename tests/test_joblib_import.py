"""Safety checks to ensure we import the real :mod:`joblib` package."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import joblib

SITE_INDICATOR = {"site-packages", "dist-packages"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def _assert_external(path: Path) -> None:
    resolved = path.resolve()
    assert any(
        part in resolved.parts for part in SITE_INDICATOR
    ), f"joblib resolved to unexpected location: {resolved!s}"
    assert (
        REPO_ROOT not in resolved.parents
    ), "joblib import should not point inside the repository"


def test_joblib_import_resolves_outside_repo() -> None:
    # Acceptance criteria: the module should come from the interpreter's
    # site-packages directory so we exercise the real dependency instead of the
    # legacy in-repo stub.  Debian/Ubuntu images sometimes use ``dist-packages``
    # instead, so we accept either spelling while still requiring an external
    # location.
    _assert_external(Path(joblib.__file__))


def test_joblib_spec_origin_outside_repo() -> None:
    spec = importlib.util.find_spec("joblib")
    assert spec is not None, "joblib should be discoverable via importlib"
    assert spec.origin, "joblib should expose a filesystem origin"
    _assert_external(Path(spec.origin))


def test_joblib_import_subprocess() -> None:
    """A fresh interpreter should resolve :mod:`joblib` from site-packages."""

    script = (
        "import pathlib, joblib; "
        "path = pathlib.Path(joblib.__file__).resolve(); "
        "print(path)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    resolved = Path(completed.stdout.strip())
    _assert_external(resolved)
