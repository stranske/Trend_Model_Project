from __future__ import annotations

import os
from pathlib import Path

import pytest

from trend_analysis.tool_layer import ToolLayer


def test_sandbox_rejects_absolute_outside(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data").mkdir(parents=True)
    monkeypatch.setenv("TREND_REPO_ROOT", str(repo_root))
    tool = ToolLayer()

    outside = tmp_path / "outside.csv"
    outside.write_text("Date,A\n2020-01-01,0.01\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sandbox"):
        tool._sandbox_path(outside)


def test_sandbox_rejects_traversal_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data").mkdir(parents=True)
    monkeypatch.setenv("TREND_REPO_ROOT", str(repo_root))
    tool = ToolLayer()

    with pytest.raises(ValueError, match="traversal"):
        tool._sandbox_path("data/../secrets.csv")


def test_sandbox_rejects_absolute_traversal_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data").mkdir(parents=True)
    monkeypatch.setenv("TREND_REPO_ROOT", str(repo_root))
    tool = ToolLayer()

    absolute_path = repo_root / "data" / ".." / "secrets.csv"
    with pytest.raises(ValueError, match="traversal"):
        tool._sandbox_path(absolute_path)


def test_sandbox_rejects_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    data_root = repo_root / "data"
    data_root.mkdir(parents=True)
    monkeypatch.setenv("TREND_REPO_ROOT", str(repo_root))
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    link_path = data_root / "linked.csv"
    target_file = outside_root / "returns.csv"
    target_file.write_text("Date,A\n2020-01-01,0.01\n", encoding="utf-8")

    try:
        os.symlink(target_file, link_path)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported in this environment")

    tool = ToolLayer()
    with pytest.raises(ValueError, match="sandbox"):
        tool._sandbox_path(link_path)
