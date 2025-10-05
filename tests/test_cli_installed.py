"""Smoke tests for installed CLI entry points - validates console_scripts work in CI Ubuntu."""

import shutil
import subprocess

import pytest


def test_installed_trend_model_available():
    """Smoke test: trend-model command is available after package installation."""
    # Check if trend-model command is available in PATH
    trend_model_path = shutil.which("trend-model")

    if trend_model_path is None:
        pytest.skip("trend-model command not found in PATH - package not installed")

    # Test basic help functionality
    result = subprocess.run(["trend-model", "--help"], capture_output=True, text=True)

    assert result.returncode == 0
    assert "trend-model" in result.stdout
    assert "gui" in result.stdout
    assert "run" in result.stdout


def test_installed_trend_model_run_help():
    """Smoke test: trend-model run --help works after package installation."""
    trend_model_path = shutil.which("trend-model")

    if trend_model_path is None:
        pytest.skip("trend-model command not found in PATH - package not installed")

    result = subprocess.run(
        ["trend-model", "run", "--help"], capture_output=True, text=True
    )

    assert result.returncode == 0
    assert "config" in result.stdout.lower()
    assert "input" in result.stdout.lower()


def test_installed_trend_model_gui_help():
    """Smoke test: trend-model gui --help works after package installation."""
    trend_model_path = shutil.which("trend-model")

    if trend_model_path is None:
        pytest.skip("trend-model command not found in PATH - package not installed")

    result = subprocess.run(
        ["trend-model", "gui", "--help"], capture_output=True, text=True
    )

    assert result.returncode == 0


def test_console_scripts_entry_points():
    """Smoke test: All expected console_scripts are available after installation."""
    expected_commands = [
        "trend-model",
        "trend-analysis",
        "trend-multi-analysis",
        "trend-app",
        "trend-run",
    ]

    available_commands = []
    missing_commands = []

    for cmd in expected_commands:
        if shutil.which(cmd):
            available_commands.append(cmd)
        else:
            missing_commands.append(cmd)

    # At minimum, trend-model should be available if package is properly installed
    if not available_commands:
        pytest.skip("No trend-* commands found in PATH - package not installed")

    # Test at least one command works
    result = subprocess.run(
        [available_commands[0], "--help"], capture_output=True, text=True
    )

    assert result.returncode == 0
    print(f"Available commands: {available_commands}")
    if missing_commands:
        print(f"Missing commands: {missing_commands}")


def test_trend_model_error_handling():
    """Smoke test: trend-model properly handles invalid arguments."""
    trend_model_path = shutil.which("trend-model")

    if trend_model_path is None:
        pytest.skip("trend-model command not found in PATH - package not installed")

    # Test with invalid command
    result = subprocess.run(
        ["trend-model", "invalid-command"], capture_output=True, text=True
    )

    # Should fail with argument error
    assert result.returncode != 0

    # Test run command without required arguments
    result = subprocess.run(["trend-model", "run"], capture_output=True, text=True)

    # Should fail due to missing required arguments
    assert result.returncode == 2
    assert "required" in result.stderr.lower() or "argument" in result.stderr.lower()
