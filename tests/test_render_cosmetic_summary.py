from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / ".github" / "scripts" / "render_cosmetic_summary.py"

if not SCRIPT_PATH.exists():
    pytest.skip(f"Script not found at {SCRIPT_PATH}, skipping tests in this module.", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("render_cosmetic_summary", SCRIPT_PATH)
_module = importlib.util.module_from_spec(_spec)
if not (_spec and _spec.loader):
    raise ImportError(f"Could not load spec or loader for {SCRIPT_PATH}")
_spec.loader.exec_module(_module)  # type: ignore[attr-defined]

build_summary_lines = _module.build_summary_lines


def test_build_summary_lines_with_changes_and_instructions() -> None:
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

    assert lines == [
        "- Status: **success**",
        "- Changed files (2):",
        "  - `alpha.py`",
        "  - `beta.py`",
        "- PR: https://example.com/pr/1",
        "- Instructions processed:",
        "  - `delete` → `foo.txt` (shellcheck)",
        "  - `format` → `bar.py`",
    ]


def test_build_summary_lines_without_changes() -> None:
    payload = {"status": "noop"}

    lines = build_summary_lines(payload)

    assert lines == [
        "- Status: **noop**",
        "- No file changes detected.",
    ]
