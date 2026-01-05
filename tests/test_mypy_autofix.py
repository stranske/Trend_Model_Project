from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.mypy_autofix as mypy_autofix
import scripts.render_mypy_summary as render_mypy_summary


def test_extract_missing_typing_symbol() -> None:
    assert (
        mypy_autofix.extract_missing_typing_symbol('Name "Optional" is not defined') == "Optional"
    )
    assert mypy_autofix.extract_missing_typing_symbol('Name "pathlib" is not defined') is None


def test_current_typing_imports_handles_multiline() -> None:
    source = """from typing import (
    Any,
    Optional,
    TypedDict,
)
"""
    assert mypy_autofix.current_typing_imports(source) == {
        "Any",
        "Optional",
        "TypedDict",
    }


def test_apply_typing_imports_inserts_new_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "example.py"
    module.write_text(
        '"""Module docstring"""\n\n'
        "from __future__ import annotations\n\n"
        "class Foo:\n"
        "    def bar(self, value: Optional[int]) -> Optional[int]:\n"
        "        return value\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mypy_autofix, "ROOT", tmp_path)

    changed = mypy_autofix.apply_typing_imports(module, {"Optional"})
    assert changed
    contents = module.read_text(encoding="utf-8")
    assert "from typing import Optional" in contents


def test_apply_typing_imports_merges_existing_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "merge_example.py"
    module.write_text(
        """from typing import Any

VALUE: Optional[int] | None = None
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(mypy_autofix, "ROOT", tmp_path)

    changed = mypy_autofix.apply_typing_imports(module, {"Optional"})
    assert changed
    contents = module.read_text(encoding="utf-8")
    assert contents.splitlines()[0] == "from typing import Any, Optional"

    # Running again should be idempotent
    changed_again = mypy_autofix.apply_typing_imports(module, {"Optional"})
    assert not changed_again


def test_gather_missing_symbols_filters_nonexistent_paths(tmp_path: Path) -> None:
    file_path = tmp_path / "needs_typing.py"
    file_path.write_text("foo: Optional[int] = 1\n", encoding="utf-8")

    diags = [
        {
            "severity": "error",
            "message": 'Name "Optional" is not defined',
            "path": str(file_path),
        },
        {
            "severity": "error",
            "message": 'Name "some_helper" is not defined',
            "path": str(file_path),
        },
    ]

    result = mypy_autofix.gather_missing_symbols(diags)
    assert result[file_path] == {"Optional"}


def test_render_mypy_summary_renders_messages(tmp_path: Path) -> None:
    diag_path = tmp_path / "diag.json"
    diag_path.write_text(
        json.dumps(
            {
                "path": "module.py",
                "line": 10,
                "column": 2,
                "severity": "error",
                "message": 'Name "Optional" is not defined',
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.txt"
    rc = render_mypy_summary.main([str(diag_path), str(report_path)])
    assert rc == 0
    assert (
        report_path.read_text(encoding="utf-8").strip()
        == 'module.py:10:2: error: Name "Optional" is not defined'
    )


def test_render_mypy_summary_success_when_empty(tmp_path: Path) -> None:
    diag_path = tmp_path / "empty.json"
    report_path = tmp_path / "report.txt"
    rc = render_mypy_summary.main([str(diag_path), str(report_path)])
    assert rc == 0
    assert report_path.read_text(encoding="utf-8").strip() == "Success: no mypy issues detected."
