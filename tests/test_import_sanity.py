"""Sanity checks for isolating the installed package from legacy modules."""
from __future__ import annotations

import os
import pkgutil
import subprocess
import sys
import textwrap
from pathlib import Path

import trend_analysis

LEGACY_MODULE_ROOTS = ("Old", "retired")


def test_trend_analysis_package_does_not_expose_legacy_modules() -> None:
    """Ensure internal package discovery never resolves legacy modules."""
    package_modules = [
        module.name
        for module in pkgutil.walk_packages(
            trend_analysis.__path__, prefix=f"{trend_analysis.__name__}."
        )
    ]

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

        import trend_analysis

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
