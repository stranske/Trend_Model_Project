"""Tests for :mod:`scripts.sync_test_dependencies`."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import sync_test_dependencies as sync


@pytest.fixture
def temp_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create an isolated repository layout for dependency syncing tests."""

    monkeypatch.chdir(tmp_path)
    # Narrow global configuration to keep the fixture deterministic.
    monkeypatch.setattr(sync, "STDLIB_MODULES", {"os", "sys"})
    monkeypatch.setattr(sync, "TEST_FRAMEWORK_MODULES", {"pytest"})
    monkeypatch.setattr(sync, "PROJECT_MODULES", {"tests", "scripts"})
    monkeypatch.setattr(sync, "MODULE_TO_PACKAGE", {"yaml": "PyYAML", "cv2": "opencv-python"})
    return tmp_path


def _create_test_files(base: Path) -> None:
    tests_dir = base / "tests"
    tests_dir.mkdir()
    tests_dir.joinpath("test_alpha.py").write_text(
        "import os\nimport yaml\nfrom pandas import DataFrame\n", encoding="utf-8"
    )
    tests_dir.joinpath("test_beta.py").write_text(
        "import cv2\ntry:\n    import requests\nexcept ImportError:\n    pass\n",
        encoding="utf-8",
    )
    cache_dir = tests_dir / "__pycache__"
    cache_dir.mkdir()
    cache_dir.joinpath("ignored.py").write_text("import should_not_count\n", encoding="utf-8")


def _write_requirements(base: Path, lines: list[str]) -> None:
    (base / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_extract_imports_handles_syntax_errors(temp_repo: Path) -> None:
    good = temp_repo / "file_good.py"
    bad = temp_repo / "file_bad.py"
    good.write_text("import yaml\nfrom pandas import Series\n", encoding="utf-8")
    bad.write_text("def broken(:\n", encoding="utf-8")

    assert sync.extract_imports_from_file(good) == {"yaml", "pandas"}
    assert sync.extract_imports_from_file(bad) == set()


def test_get_declared_dependencies_handles_missing_file(temp_repo: Path) -> None:
    deps, raw = sync.get_declared_dependencies()
    assert deps == set()
    assert raw == []


def test_get_all_test_imports_handles_missing_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert sync.get_all_test_imports() == set()


def test_find_missing_dependencies_filters_known_sets(temp_repo: Path) -> None:
    _create_test_files(temp_repo)
    _write_requirements(temp_repo, ["PyYAML==6.0", "# Test dependencies", "opencv-python==4.0"])

    missing = sync.find_missing_dependencies()
    # pandas and requests should be missing; yaml and cv2 covered via requirements/mapping.
    assert missing == {"pandas", "requests"}


def test_add_dependencies_updates_test_section(temp_repo: Path) -> None:
    _create_test_files(temp_repo)
    _write_requirements(temp_repo, ["# Base", "", "# Test dependencies", "opencv-python==4.0"])

    added = sync.add_dependencies_to_requirements({"requests", "pandas"}, fix=False)
    assert added is False

    assert sync.add_dependencies_to_requirements({"requests", "pandas"}, fix=True) is True
    new_requirements = (temp_repo / "requirements.txt").read_text(encoding="utf-8")
    assert "pandas" in new_requirements.splitlines()
    assert "requests" in new_requirements.splitlines()

    # Calling with an empty set returns immediately.
    assert sync.add_dependencies_to_requirements(set(), fix=True) is False


def test_add_dependencies_creates_section_when_missing(temp_repo: Path) -> None:
    _create_test_files(temp_repo)
    _write_requirements(temp_repo, ["dependency-one==1.0", "", "   "])

    assert sync.add_dependencies_to_requirements({"requests"}, fix=True) is True
    content = (temp_repo / "requirements.txt").read_text(encoding="utf-8")
    assert "# Test dependencies (auto-discovered)" in content
    assert content.strip().endswith("requests")


def test_main_cli_modes(temp_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _create_test_files(temp_repo)
    _write_requirements(temp_repo, ["# Test dependencies", "opencv-python==4.0"])

    monkeypatch.setattr("sys.argv", ["sync"])
    exit_code = sync.main()
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "To fix" in captured.out

    monkeypatch.setattr("sys.argv", ["sync", "--verify"])
    exit_code = sync.main()
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Found" in captured.out

    monkeypatch.setattr("sys.argv", ["sync", "--fix"])
    exit_code = sync.main()
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Added" in captured.out

    monkeypatch.setattr("sys.argv", ["sync", "--verify"])
    exit_code = sync.main()
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "All test dependencies" in captured.out

    monkeypatch.setattr("sys.argv", ["sync"])
    exit_code = sync.main()
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "All test dependencies" in captured.out


def test_module_entrypoint_executes_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "requirements.txt").write_text("# none\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["sync_deps"])

    import sys

    sys.modules.pop("scripts.sync_test_dependencies", None)

    import runpy

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("scripts.sync_test_dependencies", run_name="__main__")

    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert "All test dependencies" in captured.out
