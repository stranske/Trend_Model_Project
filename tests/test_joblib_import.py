"""Safety checks to ensure we import the real :mod:`joblib` package."""

from __future__ import annotations

from pathlib import Path

import joblib


def test_joblib_import_resolves_outside_repo() -> None:
    joblib_path = Path(joblib.__file__).resolve()
    repo_root = Path(__file__).resolve().parents[1]

    # Acceptance criteria: the module should come from the interpreter's
    # site-packages directory so we exercise the real dependency instead of the
    # legacy in-repo stub.  Debian/Ubuntu images sometimes use ``dist-packages``
    # instead, so we accept either spelling while still requiring an external
    # location.
    site_indicator = {"site-packages", "dist-packages"}
    assert any(part in joblib_path.parts for part in site_indicator), (
        f"joblib resolved to unexpected location: {joblib_path!s}"
    )
    assert repo_root not in joblib_path.parents, (
        "joblib import should not point inside the repository"
    )
