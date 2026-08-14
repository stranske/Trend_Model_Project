"""Installed-console-script contract tests for the supported CLI surface."""

from __future__ import annotations

import os
import subprocess
import venv
from pathlib import Path

import pytest

REMOVED_COMMANDS = (
    "trend-analysis",
    "trend-multi-analysis",
    "trend-model",
    "trend-app",
    "trend-run",
    "trend-quick-report",
)


@pytest.fixture()
def installed_bin(tmp_path: Path) -> Path:
    """Install this checkout into an isolated environment with shared dependencies."""
    environment = tmp_path / "installed-package"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    python = environment / "bin" / "python"
    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            str(project_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return environment / "bin"


def test_removed_compat_entrypoints_are_not_installed(installed_bin: Path) -> None:
    """Only supported console scripts are present after an isolated installation."""
    assert (installed_bin / "trend").is_file()
    assert (installed_bin / "trend-llm-proxy").is_file()
    for command in REMOVED_COMMANDS:
        assert not (installed_bin / command).exists()

    result = subprocess.run(
        [installed_bin / "trend", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{installed_bin}:{os.environ.get('PATH', '')}"},
    )
    assert result.returncode == 0, result.stderr
    assert "usage: trend" in result.stdout.lower()
