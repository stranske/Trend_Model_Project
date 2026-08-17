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
TEXT_SUFFIXES = {
    ".cfg",
    ".cjs",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}
ROOT_EXTENSIONLESS_RUNTIME_FILES = {"Dockerfile", "Makefile"}
# This is preserved review evidence, not an operator, build, or runtime surface.
ROOT_HISTORY_TEXT_FILES = {REPO_ROOT / "review-suggested-issues.md"}
ROOT_RUNTIME_TEXT_FILES = tuple(
    path
    for path in REPO_ROOT.iterdir()
    if path.is_file()
    and path not in ROOT_HISTORY_TEXT_FILES
    and (path.suffix.lower() in TEXT_SUFFIXES or path.name in ROOT_EXTENSIONLESS_RUNTIME_FILES)
)
RUNTIME_TEXT_ROOTS = (
    REPO_ROOT / "analysis",
    REPO_ROOT / "design-system",
    REPO_ROOT / "src",
    REPO_ROOT / "streamlit_app",
    REPO_ROOT / "scripts",
    REPO_ROOT / "examples",
    REPO_ROOT / "tests",
    REPO_ROOT / "docs",
    REPO_ROOT / ".github" / "workflows",
    REPO_ROOT / ".github" / "actions",
    REPO_ROOT / ".github" / "scripts",
    REPO_ROOT / "tools",
    *ROOT_RUNTIME_TEXT_FILES,
)
FORBIDDEN_RUNTIME_IMPORTS = (
    "trend." + "compat_entrypoints",
    "trend_analysis." + "cli",
    "trend_analysis." + "run_analysis",
    "trend_analysis." + "run_multi_analysis",
)
FORBIDDEN_RUNTIME_REFERENCES = FORBIDDEN_RUNTIME_IMPORTS + (
    "trend_analysis/" + "cli.py",
    "trend_analysis/" + "run_analysis.py",
    "trend_analysis/" + "run_multi_analysis.py",
)
FORBIDDEN_RUNTIME_SYMBOLS = (
    "load_market_data_" + "csv",
    "load_market_data_" + "parquet",
)
REMOVED_PATHS = (
    "src/trend/compat_entrypoints.py",
    "src/trend_analysis/" + "cli.py",
    "src/trend_analysis/" + "run_analysis.py",
    "src/trend_analysis/" + "run_multi_analysis.py",
    "src/trend_model",
    "src/trend_portfolio_app",
    "retired/trend_portfolio_app",
    "retired/tests",
    "examples/legacy_streamlit_app",
    "scripts/trend-model",
)


def _is_archived(path: Path) -> bool:
    return any(path.is_relative_to(root) for root in ARCHIVE_ROOTS)


def _include_text_file(path: Path) -> bool:
    if (
        not path.is_file()
        or "__pycache__" in path.parts
        or "node_modules" in path.parts
        or _is_archived(path)
    ):
        return False
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    return (path.parent.name == "scripts" and path.suffix == "") or (
        path.parent == REPO_ROOT and path.name in ROOT_EXTENSIONLESS_RUNTIME_FILES
    )


def _text_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if _include_text_file(root) else []
    return [path for path in root.rglob("*") if _include_text_file(path)]


def _forbidden_import_offenders(path: Path, text: str) -> list[str]:
    is_extensionless_launcher = path.suffix == "" and path.parent.name == "scripts"
    if path.suffix.lower() != ".py" and not is_extensionless_launcher:
        return []
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []
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
    try:
        display_path = path.relative_to(REPO_ROOT)
    except ValueError:
        display_path = path
    return [
        f"{display_path}: {forbidden}"
        for forbidden in FORBIDDEN_RUNTIME_IMPORTS
        if any(name == forbidden or name.startswith(f"{forbidden}.") for name in modules)
    ]


def test_removed_runtime_surfaces_do_not_return() -> None:
    """Retired packages, apps, and scripts must stay absent from the checkout."""

    returned = [path for path in REMOVED_PATHS if (REPO_ROOT / path).exists()]
    assert not returned, "Retired runtime surfaces returned:\n" + "\n".join(returned)


def test_active_runtime_and_docs_do_not_reference_removed_modules() -> None:
    """Only archived history may name the retired module entry points."""

    offenders: list[str] = []
    for root in RUNTIME_TEXT_ROOTS:
        for path in _text_files(root):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for reference in FORBIDDEN_RUNTIME_REFERENCES:
                if reference in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {reference}")
            offenders.extend(_forbidden_import_offenders(path, text))

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


def test_import_from_detection_keeps_retired_modules_absent(tmp_path: Path) -> None:
    """Multiline ``from package import name`` forms must remain forbidden."""

    candidate = tmp_path / "retired_import.py"
    candidate.write_text(
        "from trend_analysis import (\n    cli,\n)\n",
        encoding="utf-8",
    )

    offenders = _forbidden_import_offenders(candidate, candidate.read_text(encoding="utf-8"))

    assert any("trend_analysis." + "cli" in offender for offender in offenders)


def test_extensionless_launchers_remain_in_text_scan(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    launcher = scripts / "trend"
    launcher.write_text(
        "#!/usr/bin/env python\nfrom trend_analysis import cli\n",
        encoding="utf-8",
    )

    assert launcher in _text_files(tmp_path)
    offenders = _forbidden_import_offenders(launcher, launcher.read_text(encoding="utf-8"))
    assert any("trend_analysis." + "cli" in offender for offender in offenders)


def test_workflow_and_tooling_entry_points_remain_in_text_scan(tmp_path: Path) -> None:
    """Automation and repository tools are active runtime entry points too."""

    workflow = tmp_path / ".github" / "workflows" / "check.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("run: python -m trend.cli check\n", encoding="utf-8")
    tool = tmp_path / "tools" / "coverage_guard.py"
    tool.parent.mkdir()
    tool.write_text("from trend import cli\n", encoding="utf-8")
    action = tmp_path / ".github" / "actions" / "path-classifier" / "action.yml"
    action.parent.mkdir(parents=True)
    action.write_text("runs:\n  using: composite\n", encoding="utf-8")
    helper = tmp_path / ".github" / "scripts" / "issue_format.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("from trend import cli\n", encoding="utf-8")
    action_helper = tmp_path / ".github" / "actions" / "path-classifier" / "classify.js"
    action_helper.write_text("const { execFile } = require('child_process');\n", encoding="utf-8")

    assert workflow in _text_files(tmp_path / ".github" / "workflows")
    assert tool in _text_files(tmp_path / "tools")
    assert action in _text_files(tmp_path / ".github" / "actions")
    assert helper in _text_files(tmp_path / ".github" / "scripts")
    assert action_helper in _text_files(tmp_path / ".github" / "actions")


def test_root_launchers_and_active_docs_remain_in_text_scan() -> None:
    """Root launch surfaces and active Markdown must not fall outside the gate."""

    root_runtime_files = set(ROOT_RUNTIME_TEXT_FILES)
    expected_launchers = {
        REPO_ROOT / "Dockerfile",
        REPO_ROOT / "docker-compose.yml",
        REPO_ROOT / "Makefile",
    }
    expected_active_docs = set(REPO_ROOT.glob("*.md")) - ROOT_HISTORY_TEXT_FILES

    assert expected_launchers <= root_runtime_files
    assert expected_active_docs <= root_runtime_files


def test_legacy_surface_ci_runs_for_every_classified_change() -> None:
    """Every scanned surface must trigger the CI job that enforces this contract."""

    gate = (REPO_ROOT / ".github" / "workflows" / "pr-00-gate.yml").read_text(encoding="utf-8")
    legacy_job_header = gate.split("  legacy-surface:\n", 1)[1].split("    runs-on:", 1)[0]

    assert "needs.detect.result == 'success'" in legacy_job_header
    assert "is_python_code" not in legacy_job_header
    assert "is_docs_only" not in legacy_job_header


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
