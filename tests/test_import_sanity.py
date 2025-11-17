"""Sanity checks for isolating the installed package from legacy modules."""
from __future__ import annotations

import importlib
import os
import pkgutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

LEGACY_MODULE_ROOTS = ("Old", "retired")
PACKAGE_NAMES = ("trend_analysis", "trend_model", "trend_portfolio_app")


def _collect_package_modules(package_name: str) -> list[str]:
    """Return the fully-qualified module names for a package tree."""

    package = importlib.import_module(package_name)
    return [
        module.name
        for module in pkgutil.walk_packages(
            package.__path__, prefix=f"{package.__name__}."
        )
    ]


@pytest.mark.parametrize("package_name", PACKAGE_NAMES)
def test_packages_do_not_expose_legacy_modules(package_name: str) -> None:
    """Ensure internal package discovery never resolves legacy modules."""
    package_modules = _collect_package_modules(package_name)

    for legacy in LEGACY_MODULE_ROOTS:
        forbidden_fragment = f".{legacy}."
        assert all(
            forbidden_fragment not in module_name for module_name in package_modules
        ), forbidden_fragment


def test_src_only_import_rejects_legacy_modules(tmp_path: Path) -> None:
    """Editable installs should not leak top-level legacy folders into imports."""
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(src_dir)
        if not pythonpath
        else os.pathsep.join((str(src_dir), pythonpath))
    )

    script = textwrap.dedent(
        f"""
        import importlib

        for package_name in {PACKAGE_NAMES!r}:
            importlib.import_module(package_name)

        failures = []
        for module_name in {LEGACY_MODULE_ROOTS!r}:
            try:
                importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
            failures.append(module_name)

        if failures:
            raise SystemExit("Unexpectedly imported legacy modules: " + str(failures))
        """
    )

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        check=True,
    )
