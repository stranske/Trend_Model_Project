"""Verify issue #6205 and #6208 acceptance evidence and copy-exclusion probe."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import tests.test_cli_installed as cli_installed
from scripts.runner_lib.acceptance_gate import extract_commands

ROOT = Path(__file__).resolve().parents[2]
ACCEPTANCE = ROOT / "docs/acceptance/issue-6205.md"


def test_acceptance_record_extracts_passing_pytest_gates() -> None:
    """Ensure the acceptance record exposes the expected runnable pytest gates."""
    gates = extract_commands(ACCEPTANCE.read_text(encoding="utf-8"))

    assert [(gate.criterion, gate.command) for gate in gates] == [
        (0, "python -m pytest -q tests/test_cli_installed.py -k install_source_copy"),
        (
            1,
            "python -m pytest -q "
            "tests/test_cli_installed.py::test_install_source_copy_excludes_disappearing_demo_marker",
        ),
    ]


def test_acceptance_record_documents_complete_supported_runtime_evidence() -> None:
    """Confirm docs/acceptance/issue-6205.md records all required receipts and no open checkboxes."""
    markdown = ACCEPTANCE.read_text(encoding="utf-8")

    required_fragments = (
        "f77623978ffafaeaa5f40b486717063bd1bd1a36",
        "cde9f21439ba2f72ca596f54cd017c7673c6221b",
        "0b0348f8fea59683199d5a08a4f2d4ade1faf680",
        "Python 3.14.10",
        "python -m pytest -q tests/test_cli_installed.py -k install_source_copy",
        "python -m pytest -q tests/test_cli_installed.py::test_install_source_copy_excludes_disappearing_demo_marker",
        "shutil.Error",
        "./scripts/run_tests.sh",
        "7064 passed, 11 skipped",
        "gh run view 42518199588 --repo stranske/Trend_Model_Project --json headSha,attempt,conclusion,jobs",
        "128848973787",
        "20261108T202925Z",
        "ready-for-maint71",
    )
    for fragment in required_fragments:
        assert fragment in markdown, f"Missing required receipt fragment: {fragment}"

    assert "\n- [ ] " not in markdown


def test_removing_fast_demo_mode_exclusion_triggers_named_shutil_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify the regression test fails without .fast_demo_mode exclusion and passes when restored."""
    without_marker_exclusion = shutil.ignore_patterns(
        ".env",
        ".env.*",
        "secrets.toml",
        ".git",
        ".github",
        ".venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".workflows-lib",
        "__pycache__",
        "*.pyc",
        "node_modules",
        "archives",
        "notebooks",
        "build",
        "dist",
        "*.egg-info",
    )
    with monkeypatch.context() as broken_ctx:
        broken_ctx.setattr(cli_installed, "_INSTALL_COPY_IGNORE", without_marker_exclusion)
        with pytest.raises(shutil.Error, match=r"\.fast_demo_mode"):
            cli_installed.test_install_source_copy_excludes_disappearing_demo_marker(
                tmp_path / "broken", broken_ctx
            )

    cli_installed.test_install_source_copy_excludes_disappearing_demo_marker(
        tmp_path / "restored", monkeypatch
    )
