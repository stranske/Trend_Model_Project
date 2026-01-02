"""Safety checks to ensure we import the real :mod:`joblib` package."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

SITE_INDICATOR = {"site-packages", "dist-packages"}
REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_INTERNAL_PREFIXES = {REPO_ROOT / name for name in (".venv", "venv", ".tox")}

joblib = pytest.importorskip("joblib")

ENTRYPOINT_MODULES = (
    "trend.cli",
    "trend_analysis.cli",
    "trend_analysis.run_analysis",
    "trend_analysis.run_multi_analysis",
)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _assert_external(path: Path) -> None:
    resolved = path.resolve()
    assert any(part in resolved.parts for part in SITE_INDICATOR), (
        f"joblib resolved to unexpected location: {resolved!s}"
    )
    if REPO_ROOT in resolved.parents:
        if any(_is_under(resolved, prefix) for prefix in ALLOWED_INTERNAL_PREFIXES):
            return
        raise AssertionError("joblib import should not point inside the repository")


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


@pytest.mark.parametrize("module_name", ENTRYPOINT_MODULES)
def test_cli_entrypoints_expose_help_and_external_joblib(module_name: str) -> None:
    module = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    assert main is not None, f"Module '{module_name}' does not expose a 'main' function"
    try:
        main(["--help"])
    except SystemExit as exc:  # argparse exits after printing help
        assert exc.code == 0
    _assert_external(Path(joblib.__file__))
