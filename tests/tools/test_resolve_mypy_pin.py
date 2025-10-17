from __future__ import annotations

from pathlib import Path

import pytest

from tools import resolve_mypy_pin


def test_resolve_pin_prefers_pyproject_pin(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
        [tool.mypy]
        python_version = "3.12"
        """,
        encoding="utf-8",
    )

    result = resolve_mypy_pin.resolve_pin(pyproject, "3.11")

    assert result.pin == "3.12"
    assert result.notices == ()


def test_resolve_pin_defaults_to_matrix_when_missing_file(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"

    result = resolve_mypy_pin.resolve_pin(pyproject, " 3.11 ")

    assert result.pin == "3.11"
    assert result.notices == (
        (
            "notice",
            "pyproject.toml not found; defaulting mypy python_version to matrix interpreter 3.11",
        ),
    )


def test_resolve_pin_warns_without_pyproject_and_matrix(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"

    result = resolve_mypy_pin.resolve_pin(pyproject, None)

    assert result.pin is None
    assert result.notices == (
        (
            "warning",
            "pyproject.toml not found and no matrix interpreter provided; skipping mypy pin resolution",
        ),
    )


def test_resolve_pin_warns_when_no_pin_available(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
        [tool]
        other = "value"
        """,
        encoding="utf-8",
    )

    result = resolve_mypy_pin.resolve_pin(pyproject, "")

    assert result.pin is None
    assert (
        "warning",
        "No mypy python_version pin found and matrix version unavailable",
    ) in result.notices


def test_main_writes_to_github_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
        [tool.mypy]
        python_version = "3.12"
        """,
        encoding="utf-8",
    )

    output_file = tmp_path / "output.txt"
    monkeypatch.setenv("PYPROJECT_PATH", str(pyproject))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.delenv("MATRIX_PYTHON_VERSION", raising=False)

    return_code = resolve_mypy_pin.main(())

    assert return_code == 0
    captured = capsys.readouterr()
    assert "Resolved mypy python_version pin: 3.12" in captured.out
    assert output_file.read_text(encoding="utf-8").strip() == "python-version=3.12"


def test_main_reports_toml_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("invalid = [unclosed", encoding="utf-8")

    monkeypatch.setenv("PYPROJECT_PATH", str(pyproject))
    monkeypatch.setenv("MATRIX_PYTHON_VERSION", "3.11")

    return_code = resolve_mypy_pin.main(())

    assert return_code == 1
    captured = capsys.readouterr()
    assert f"::error file={pyproject}" in captured.out
    assert "Failed to parse" in captured.out
