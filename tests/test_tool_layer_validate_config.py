from __future__ import annotations

from pathlib import Path

import pytest

from trend_analysis.config.validation import ValidationResult
from trend_analysis.tool_layer import ToolLayer


def test_validate_config_reports_missing_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    config_root = repo_root / "config"
    config_root.mkdir(parents=True)
    monkeypatch.setenv("TREND_REPO_ROOT", str(repo_root))
    tool = ToolLayer()

    result = tool.validate_config({}, base_path=config_root)

    assert result.status == "success"
    assert isinstance(result.data, ValidationResult)
    assert result.data.valid is False
    assert result.data.errors


def test_validate_config_rejects_paths_outside_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "config").mkdir(parents=True)
    monkeypatch.setenv("TREND_REPO_ROOT", str(repo_root))
    tool = ToolLayer()

    result = tool.validate_config({}, base_path=tmp_path / "outside")

    assert result.status == "error"
    assert "sandbox" in (result.message or "")


def test_validate_config_rejects_path_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "config").mkdir(parents=True)
    monkeypatch.setenv("TREND_REPO_ROOT", str(repo_root))
    tool = ToolLayer()

    result = tool.validate_config({}, base_path=Path("config/../secrets"))

    assert result.status == "error"
    assert (result.message or "").startswith("SecurityError: Path traversal detected:")


def test_validate_config_rejects_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    config_root = repo_root / "config"
    config_root.mkdir(parents=True)
    monkeypatch.setenv("TREND_REPO_ROOT", str(repo_root))
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    link_path = config_root / "linked"

    try:
        link_path.symlink_to(outside_root)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported in this environment")

    tool = ToolLayer()
    result = tool.validate_config({}, base_path=link_path)

    assert result.status == "error"
    assert "sandbox" in (result.message or "")
