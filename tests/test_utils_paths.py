from __future__ import annotations

from pathlib import Path

from utils.paths import proj_path


def test_proj_path_defaults_to_repo_root() -> None:
    root = proj_path()
    assert (root / "pyproject.toml").exists()


def test_proj_path_honours_env_override(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "override"
    override.mkdir()
    monkeypatch.setenv("TREND_REPO_ROOT", str(override))

    assert proj_path() == override.resolve()
    assert proj_path("nested", "file.txt") == override / "nested" / "file.txt"


def test_proj_path_stable_across_cwd(monkeypatch, tmp_path: Path) -> None:
    root = proj_path()
    monkeypatch.chdir(tmp_path)

    assert proj_path() == root
    assert proj_path("src").parent == root
