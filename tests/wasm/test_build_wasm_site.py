"""Tests for the GitHub Pages stlite artifact assembler."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_wasm_site import _validated_relative_path, assemble_site


def _write(path: Path, content: str = "fixture") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_assemble_site_copies_runtime_manifest_and_declared_sources(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    output = tmp_path / "site"
    _write(repo / "demo/wasm/index.html", "<main>demo</main>")
    _write(repo / "demo/wasm/vendor/stlite.js")
    _write(repo / "streamlit_app/app.py", "print('demo')")
    _write(repo / "demo/demo_returns.csv", "date,value\n2026-01-01,1\n")

    manifest = assemble_site(repo_root=repo, output_dir=output)

    assert (output / "index.html").read_text(encoding="utf-8") == "<main>demo</main>"
    assert (output / "vendor/stlite.js").is_file()
    assert (output / "app/streamlit_app/app.py").is_file()
    assert (output / "app/demo/demo_returns.csv").is_file()
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest


@pytest.mark.parametrize("value", ["../secret", "/absolute", "a/../../secret", "./relative"])
def test_manifest_paths_cannot_escape_the_pages_artifact(value: str) -> None:
    with pytest.raises(ValueError, match="unsafe manifest path"):
        _validated_relative_path(value)


def test_assemble_site_refuses_repository_root_as_output(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    with pytest.raises(ValueError, match="refusing broad output path"):
        assemble_site(repo_root=repo, output_dir=repo)
