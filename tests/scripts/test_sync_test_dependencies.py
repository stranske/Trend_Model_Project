"""Unit tests for the pyproject-based test dependency synchroniser."""

from __future__ import annotations

import runpy
import sys
import textwrap
from pathlib import Path

import pytest

from scripts import sync_test_dependencies as sync


def _write_pyproject(base: Path, dev: list[str] | None = None) -> None:
    dev = dev or []
    content = textwrap.dedent(
        """
        [project]
        name = "demo"
        version = "0.0.0"
        dependencies = [
            "PyYAML",
            "opencv-python",
        ]

        [project.optional-dependencies]
        dev = [
        {dev_entries}
        ]
        """
    ).strip()
    dev_block = "\n".join(f"    \"{item}\"," for item in dev)
    content = content.format(dev_entries=dev_block)
    base.joinpath("pyproject.toml").write_text(content + "\n", encoding="utf-8")


@pytest.fixture
def temp_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create an isolated repository layout for dependency syncing tests."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sync, "STDLIB_MODULES", {"os", "sys"})
    monkeypatch.setattr(sync, "TEST_FRAMEWORK_MODULES", {"pytest"})
    monkeypatch.setattr(sync, "PROJECT_MODULES", {"tests", "scripts"})
    monkeypatch.setattr(
        sync, "MODULE_TO_PACKAGE", {"yaml": "PyYAML", "cv2": "opencv-python"}
    )

    _write_pyproject(tmp_path, dev=["pytest", "tomlkit"])
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
    cache_dir.joinpath("ignored.py").write_text(
        "import should_not_count\n", encoding="utf-8"
    )


def test_extract_imports_handles_syntax_errors(temp_repo: Path) -> None:
    good = temp_repo / "file_good.py"
    bad = temp_repo / "file_bad.py"
    good.write_text("import yaml\nfrom pandas import Series\n", encoding="utf-8")
    bad.write_text("def broken(:\n", encoding="utf-8")

    assert sync.extract_imports_from_file(good) == {"yaml", "pandas"}
    assert sync.extract_imports_from_file(bad) == set()


def test_get_declared_dependencies_handles_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    deps, groups = sync.get_declared_dependencies()
    assert deps == set()
    assert groups == {}


def test_find_missing_dependencies_filters_known_sets(temp_repo: Path) -> None:
    _create_test_files(temp_repo)
    missing = sync.find_missing_dependencies()
    # pandas and requests should be missing; yaml and cv2 are covered.
    assert missing == {"pandas", "requests"}


def test_add_dependencies_updates_dev_extra(temp_repo: Path) -> None:
    _create_test_files(temp_repo)
    missing = {"pandas", "requests"}

    # Dry run returns False
    assert sync.add_dependencies_to_pyproject(missing, fix=False) is False

    assert sync.add_dependencies_to_pyproject(missing, fix=True) is True
    content = (temp_repo / "pyproject.toml").read_text(encoding="utf-8")
    assert "pandas" in content
    assert "requests" in content

    # Calling with an empty set returns immediately.
    assert sync.add_dependencies_to_pyproject(set(), fix=True) is False


def test_main_cli_modes(
    temp_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _create_test_files(temp_repo)

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
    assert "Added dependencies" in captured.out

    monkeypatch.setattr("sys.argv", ["sync", "--verify"])
    exit_code = sync.main()
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "All test dependencies" in captured.out

    monkeypatch.setattr("sys.argv", ["sync"])
    exit_code = sync.main()
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "All test dependencies" in captured.out


def test_module_entrypoint_executes_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_pyproject(tmp_path, dev=["pytest", "tomlkit"])
    (tmp_path / "tests").mkdir()
    monkeypatch.setattr("sys.argv", ["sync_deps"])

    sys.modules.pop("scripts.sync_test_dependencies", None)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("scripts.sync_test_dependencies", run_name="__main__")

    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert "All test dependencies" in captured.out
