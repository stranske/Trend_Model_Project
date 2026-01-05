"""Tests for resolve_mypy_pin module.

Tests the utility for resolving which Python version should run mypy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import resolve_mypy_pin


def test_get_mypy_python_version_returns_none_when_no_pyproject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test returns None when pyproject.toml doesn't exist."""
    monkeypatch.chdir(tmp_path)
    result = resolve_mypy_pin.get_mypy_python_version()
    assert result is None


def test_get_mypy_python_version_returns_none_when_no_mypy_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test returns None when [tool.mypy] section is missing."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
        [tool]
        other = "value"
        """,
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    result = resolve_mypy_pin.get_mypy_python_version()
    assert result is None


def test_get_mypy_python_version_extracts_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test extracts python_version from [tool.mypy] section."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
        [tool.mypy]
        python_version = "3.12"
        strict = true
        """,
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    result = resolve_mypy_pin.get_mypy_python_version()
    assert result == "3.12"


def test_get_mypy_python_version_handles_unquoted_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test handles unquoted version numbers (some TOML styles)."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
        [tool.mypy]
        python_version = 3.11
        """,
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    result = resolve_mypy_pin.get_mypy_python_version()
    # Should work with regex fallback even if tomlkit not available
    assert result is not None


def test_main_writes_to_github_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test main writes resolved version to GITHUB_OUTPUT."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
        [tool.mypy]
        python_version = "3.12"
        """,
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    output_file = tmp_path / "output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.delenv("MATRIX_PYTHON_VERSION", raising=False)

    return_code = resolve_mypy_pin.main()

    assert return_code == 0
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "python-version=3.12" in content


def test_main_uses_matrix_version_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test main falls back to MATRIX_PYTHON_VERSION when no mypy config."""
    monkeypatch.chdir(tmp_path)  # No pyproject.toml

    output_file = tmp_path / "output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setenv("MATRIX_PYTHON_VERSION", "3.11")

    return_code = resolve_mypy_pin.main()

    assert return_code == 0
    content = output_file.read_text(encoding="utf-8")
    assert "python-version=3.11" in content


def test_main_defaults_to_311_without_any_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test main defaults to 3.11 when nothing is configured."""
    monkeypatch.chdir(tmp_path)

    output_file = tmp_path / "output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.delenv("MATRIX_PYTHON_VERSION", raising=False)

    return_code = resolve_mypy_pin.main()

    assert return_code == 0
    content = output_file.read_text(encoding="utf-8")
    assert "python-version=3.11" in content
