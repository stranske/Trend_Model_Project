"""Tests for the pyproject-only tool version synchroniser."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from scripts import sync_tool_versions as sync


def _minimal_config() -> sync.ToolConfig:
    """Return a compact configuration used within the tests."""

    return sync.ToolConfig(
        env_key="BLACK_VERSION",
        package_name="black",
        pyproject_pattern=sync._compile(r'"black==(?P<version>[^\"]+)"'),
        pyproject_format='"black=={version}",',
    )


def _configure_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the module globals to a temporary repository layout."""

    pin_file = tmp_path / "pins.env"
    pyproject = tmp_path / "pyproject.toml"

    monkeypatch.setattr(sync, "PIN_FILE", pin_file)
    monkeypatch.setattr(sync, "PYPROJECT_FILE", pyproject)
    monkeypatch.setattr(sync, "TOOL_CONFIGS", (_minimal_config(),))


def _write_repo_files(tmp_path: Path, version: str = "23.0") -> None:
    """Create pin and pyproject files for the fake repo."""

    (tmp_path / "pins.env").write_text(
        f"# comment\nBLACK_VERSION={version}\nMALFORMED_LINE\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        '[tool.example]\nrequires = [\n    "black==22.0",\n]\n',
        encoding="utf-8",
    )


def test_parse_env_file_validates_presence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_repo(tmp_path, monkeypatch)
    _write_repo_files(tmp_path)

    values = sync.parse_env_file(sync.PIN_FILE)
    assert values == {"BLACK_VERSION": "23.0"}

    sync.PIN_FILE.write_text("# comment only\n", encoding="utf-8")
    with pytest.raises(sync.SyncError, match="missing keys"):
        sync.parse_env_file(sync.PIN_FILE)

    with pytest.raises(sync.SyncError, match="does not exist"):
        sync.parse_env_file(sync.PIN_FILE.parent / "missing.env")


def test_ensure_pyproject_reports_and_applies_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_repo(tmp_path, monkeypatch)
    _write_repo_files(tmp_path)
    env = {"BLACK_VERSION": "23.0"}

    content = sync.PYPROJECT_FILE.read_text(encoding="utf-8")
    updated, mismatches = sync.ensure_pyproject(
        content, sync.TOOL_CONFIGS, env, apply=False
    )
    assert updated == content
    assert mismatches == {"black": "pyproject has 22.0, pin file requires 23.0"}

    updated, mismatches = sync.ensure_pyproject(
        content, sync.TOOL_CONFIGS, env, apply=True
    )
    assert "black==23.0" in updated
    assert mismatches == {"black": "pyproject has 22.0, pin file requires 23.0"}

    with pytest.raises(sync.SyncError, match="missing an entry"):
        sync.ensure_pyproject("[tool]\n", sync.TOOL_CONFIGS, env, apply=False)


def test_main_check_and_apply_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _configure_repo(tmp_path, monkeypatch)
    _write_repo_files(tmp_path)

    # First run: mismatch surfaces and exits with status 1 in check mode.
    exit_code = sync.main(["--check"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Use --apply" in captured.err

    # Apply mode rewrites files and succeeds.
    exit_code = sync.main(["--apply"])
    assert exit_code == 0
    assert "black==23.0" in sync.PYPROJECT_FILE.read_text(encoding="utf-8")

    # Second check run succeeds with aligned versions.
    exit_code = sync.main(["--check"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Use --apply" not in captured.err

    # Running apply again without changes keeps files untouched but exercises the branch.
    exit_code = sync.main(["--apply"])
    assert exit_code == 0

    # Mutually exclusive arguments trigger argparse error.
    with pytest.raises(SystemExit):
        sync.main(["--check", "--apply"])


def test_module_entrypoint_reports_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["sync_tool_versions"])

    import sys

    sys.modules.pop("scripts.sync_tool_versions", None)

    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "autofix-versions.env").write_text("# empty\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[tool.example]\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("scripts.sync_tool_versions", run_name="__main__")

    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "Pin file" in captured.err
