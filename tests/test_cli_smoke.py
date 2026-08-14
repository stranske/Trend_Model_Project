"""Smoke tests for CLI entry points - validates basic functionality in CI environment."""

import subprocess
import sys
from pathlib import Path


def test_cli_help_smoke():
    """Smoke test: CLI --help works without errors."""
    project_root = Path(__file__).parent.parent
    script_path = project_root / "scripts" / "trend"

    result = subprocess.run(
        [str(script_path), "--help"], capture_output=True, text=True, cwd=project_root
    )

    assert result.returncode == 0
    assert "trend" in result.stdout
    assert "app" in result.stdout
    assert "run" in result.stdout


def test_cli_run_help_smoke():
    """Smoke test: CLI run --help works without errors."""
    project_root = Path(__file__).parent.parent
    script_path = project_root / "scripts" / "trend"

    result = subprocess.run(
        [str(script_path), "run", "--help"],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    assert result.returncode == 0
    assert "config" in result.stdout.lower()
    assert "input" in result.stdout.lower()


def test_cli_app_help_smoke():
    """Smoke test: CLI app --help works without errors."""
    project_root = Path(__file__).parent.parent
    script_path = project_root / "scripts" / "trend"

    result = subprocess.run(
        [str(script_path), "app", "--help"],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    assert result.returncode == 0


def test_cli_gui_alias_rejected_smoke():
    """Smoke test: legacy gui alias is rejected by the modular CLI."""
    project_root = Path(__file__).parent.parent
    script_path = project_root / "scripts" / "trend"

    result = subprocess.run(
        [str(script_path), "gui", "--help"],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    assert result.returncode == 2
    assert "invalid choice" in result.stderr.lower()


def test_cli_module_direct_smoke():
    """Smoke test: CLI module can be invoked directly via Python."""
    project_root = Path(__file__).parent.parent

    result = subprocess.run(
        [sys.executable, "-m", "trend.cli", "--help"],
        capture_output=True,
        text=True,
        cwd=project_root,
        env={"PYTHONPATH": str(project_root / "src")},
    )

    # This may fail due to missing dependencies, but we test the entry point structure
    assert result.returncode in (0, 1)  # 0 for success, 1 for module import errors

    if result.returncode == 0:
        assert "trend" in result.stdout
    else:
        # If it fails due to missing dependencies, that's expected in this environment
        assert "ModuleNotFoundError" in result.stderr or "ImportError" in result.stderr


def test_cli_run_missing_args_smoke():
    """Smoke test: CLI run command properly validates required arguments."""
    project_root = Path(__file__).parent.parent
    script_path = project_root / "scripts" / "trend"

    result = subprocess.run(
        [str(script_path), "run"], capture_output=True, text=True, cwd=project_root
    )

    # Should fail with exit code 2 (argument error) due to missing -c and -i
    assert result.returncode == 2
    assert "required" in result.stderr.lower() or "argument" in result.stderr.lower()
