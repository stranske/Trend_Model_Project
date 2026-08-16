"""Final regression gates for retired runtime surfaces and supported commands."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOTS = (
    REPO_ROOT / "archives",
    REPO_ROOT / "docs" / "archive",
    REPO_ROOT / "docs" / "keepalive",
    REPO_ROOT / "notebooks" / "old",
)
RUNTIME_TEXT_ROOTS = (
    REPO_ROOT / "src",
    REPO_ROOT / "streamlit_app",
    REPO_ROOT / "scripts",
    REPO_ROOT / "docs",
    REPO_ROOT / "pyproject.toml",
)
FORBIDDEN_RUNTIME_IMPORTS = (
    "trend." + "compat_entrypoints",
    "trend_analysis." + "run_analysis",
    "trend_analysis." + "run_multi_analysis",
)
FORBIDDEN_RUNTIME_SYMBOLS = (
    "load_market_data_" + "csv",
    "load_market_data_" + "parquet",
)
REMOVED_PATHS = (
    "src/trend/compat_entrypoints.py",
    "src/trend_analysis/run_analysis.py",
    "src/trend_analysis/run_multi_analysis.py",
    "src/trend_model",
    "src/trend_portfolio_app",
    "retired/trend_portfolio_app",
    "retired/tests",
    "examples/legacy_streamlit_app",
    "scripts/trend-model",
)


def _is_archived(path: Path) -> bool:
    return any(path.is_relative_to(root) for root in ARCHIVE_ROOTS)


def _text_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and not _is_archived(path)
    ]


def test_removed_runtime_paths_do_not_return() -> None:
    """Retired packages, apps, and scripts must stay absent from the checkout."""

    returned = [path for path in REMOVED_PATHS if (REPO_ROOT / path).exists()]
    assert not returned, "Retired runtime surfaces returned:\n" + "\n".join(returned)


def test_active_runtime_and_docs_do_not_reference_removed_modules() -> None:
    """Only archived history may name the retired module entry points."""

    offenders: list[str] = []
    for root in RUNTIME_TEXT_ROOTS:
        for path in _text_files(root):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for module in FORBIDDEN_RUNTIME_IMPORTS:
                if module in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {module}")

    assert not offenders, "Active surfaces reference retired modules:\n" + "\n".join(offenders)


def test_active_runtime_does_not_restore_removed_data_loaders() -> None:
    """Canonical data loading must not be shadowed by compatibility shims."""

    offenders: list[str] = []
    for root in RUNTIME_TEXT_ROOTS:
        for path in _text_files(root):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for symbol in FORBIDDEN_RUNTIME_SYMBOLS:
                if symbol in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {symbol}")

    assert not offenders, "Active surfaces restore removed data loaders:\n" + "\n".join(offenders)


def test_tests_do_not_import_retired_runtime_modules() -> None:
    """Test names may mention removals, but no test may import a retired module."""

    offenders: list[str] = []
    for path in (REPO_ROOT / "tests").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        modules.extend(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        modules.extend(
            f"{node.module}.{alias.name}"
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
            for alias in node.names
        )
        for module in FORBIDDEN_RUNTIME_IMPORTS:
            if any(name == module or name.startswith(f"{module}.") for name in modules):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {module}")

    assert not offenders, "Tests import retired modules:\n" + "\n".join(offenders)


def test_import_from_detection_keeps_retired_modules_absent(tmp_path: Path) -> None:
    """Qualified names from ``from package import name`` must remain forbidden."""

    candidate = tmp_path / "retired_import.py"
    candidate.write_text("from trend import compat_entrypoints\n", encoding="utf-8")
    tree = ast.parse(candidate.read_text(encoding="utf-8"), filename=str(candidate))
    modules = [
        f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
        for alias in node.names
    ]
    offenders = [
        forbidden
        for forbidden in FORBIDDEN_RUNTIME_IMPORTS
        if any(name == forbidden or name.startswith(f"{forbidden}.") for name in modules)
    ]
    assert "trend.compat_entrypoints" in offenders


@pytest.mark.parametrize(
    ("arguments", "expected_output"),
    [
        (("--help",), "usage: trend"),
        (("run", "--help"), "usage: trend run"),
        (("report", "--help"), "usage: trend report"),
        (("quick-report", "--help"), "usage: trend quick-report"),
        (("app", "--help"), "usage: trend app"),
        (("check", "--help"), "usage: trend check"),
        (("mc", "list"), "hf_macro_20y"),
        (("mc", "validate", "--help"), "usage: trend mc validate"),
        (("mc", "run", "--help"), "usage: trend mc run"),
        (("mc", "viz", "--help"), "usage: trend mc viz"),
    ],
)
def test_supported_cli_surface_smoke(arguments: tuple[str, ...], expected_output: str) -> None:
    """The final command-tree smoke covers every supported public CLI surface."""

    result = subprocess.run(
        [sys.executable, "-m", "trend.cli", *arguments],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert expected_output in result.stdout.lower()
