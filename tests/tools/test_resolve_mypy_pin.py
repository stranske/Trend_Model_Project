"""Tests for resolve_mypy_pin module."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import resolve_mypy_pin


def test_get_mypy_python_version_from_pyproject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test extracting python_version from pyproject.toml."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.mypy]
python_version = "3.12"
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = resolve_mypy_pin.get_mypy_python_version()

    assert result == "3.12"


def test_get_mypy_python_version_returns_none_when_missing(
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
[tool.black]
line-length = 100
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = resolve_mypy_pin.get_mypy_python_version()

    assert result is None


def test_main_writes_to_github_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test main() writes resolved version to GITHUB_OUTPUT."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.mypy]
python_version = "3.12"
""",
        encoding="utf-8",
    )
    output_file = tmp_path / "output.txt"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.delenv("MATRIX_PYTHON_VERSION", raising=False)

    result = resolve_mypy_pin.main()

    assert result == 0
    assert "python-version=3.12" in output_file.read_text(encoding="utf-8")


def test_main_uses_matrix_version_as_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test main() falls back to MATRIX_PYTHON_VERSION when no pyproject.toml."""
    output_file = tmp_path / "output.txt"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setenv("MATRIX_PYTHON_VERSION", "3.11")

    result = resolve_mypy_pin.main()

    assert result == 0
    assert "python-version=3.11" in output_file.read_text(encoding="utf-8")


def test_main_defaults_to_311_when_no_version_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test main() defaults to 3.11 when no version is available."""
    output_file = tmp_path / "output.txt"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.delenv("MATRIX_PYTHON_VERSION", raising=False)

    result = resolve_mypy_pin.main()

    assert result == 0
    assert "python-version=3.11" in output_file.read_text(encoding="utf-8")
