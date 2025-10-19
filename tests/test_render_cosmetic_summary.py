from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / ".github" / "scripts" / "render_cosmetic_summary.py"

if not SCRIPT_PATH.exists():
    pytest.skip(
        f"Script not found at {SCRIPT_PATH}, skipping tests in this module.",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def build_summary_lines() -> Callable[[dict[str, Any]], list[str]]:
    spec = importlib.util.spec_from_file_location("render_cosmetic_summary", SCRIPT_PATH)
    if not (spec and spec.loader):
        raise ImportError(f"Could not load spec or loader for {SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module.build_summary_lines


def test_build_summary_lines_with_changes_and_instructions(build_summary_lines) -> None:
    payload = {
        "status": "success",
        "changed_files": ["alpha.py", "beta.py"],
        "pr_url": "https://example.com/pr/1",
        "instructions": [
            {"kind": "delete", "path": "foo.txt", "guard": "shellcheck"},
            {"kind": "format", "path": "bar.py"},
        ],
    }

    lines = build_summary_lines(payload)

    assert lines[0] == "- Status: **success**"
    assert "Changed files" in lines[1] and "2" in lines[1]
    assert any("alpha.py" in line for line in lines)
    assert any("beta.py" in line for line in lines)
    assert any(line.startswith("- PR:") and "https://example.com/pr/1" in line for line in lines)
    assert any("Instructions processed" in line for line in lines)
    assert any("delete" in line and "foo.txt" in line for line in lines)
    assert any("format" in line and "bar.py" in line for line in lines)


def test_build_summary_lines_without_changes(build_summary_lines) -> None:
    payload = {"status": "noop"}

    lines = build_summary_lines(payload)

    assert lines[0] == "- Status: **noop**"
    assert any("No file changes" in line for line in lines)
