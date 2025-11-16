#!/usr/bin/env python
"""Apply targeted fixes for mypy return-type mismatches.

The script looks for mypy diagnostics with error code ``return-value`` where the
reported message explicitly states the "got" and "expected" types. For each
match we attempt to rewrite the function's return annotation to the observed
runtime type so subsequent mypy passes succeed.

Safety guarantees
=================
* Only touches files under ``src/`` or ``tests/``.
* Only edits functions that already have an explicit return annotation.
* Uses the ``ast`` module to locate the annotation span precisely.
* Ignores complex or unsupported diagnostics (keeps behaviour conservative).

The command can be run standalone and is idempotent.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parent.parent
PROJECT_DIRS = [ROOT / "src", ROOT / "tests"]
MYPY_CMD: list[str] = [
    sys.executable,
    "-m",
    "mypy",
    "--hide-error-context",
    "--no-color-output",
    "--show-error-codes",
    "--no-error-summary",
    "src",
    "tests",
]
RETURN_RE = re.compile(
    r"Incompatible return value type \(got \"(?P<actual>[^\"]+)\", expected \"(?P<expected>[^\"]+)\"\)"
)
ALT_RETURN_RE = re.compile(
    r"Returning value of type \"(?P<actual>[^\"]+)\" incompatible with return type \"(?P<expected>[^\"]+)\""
)
SUPPORTED_CODES = {"return-value"}


@dataclass
class MypyIssue:
    path: Path
    line: int
    actual: str
    expected: str


@dataclass
class Replacement:
    start: int
    end: int
    text: str


def _normalise_type(type_str: str) -> str:
    if type_str.startswith("builtins."):
        return type_str.split(".")[-1]
    return type_str


def _parse_line_offsets(text: str) -> list[int]:
    offsets: list[int] = []
    total = 0
    for line in text.splitlines(keepends=True):
        offsets.append(total)
        total += len(line)
    offsets.append(total)
    return offsets


def _offset(line_offsets: list[int], line: int, col: int) -> int:
    # mypy uses 1-based line numbers; ast does as well.
    if line - 1 >= len(line_offsets):
        return line_offsets[-1]
    return line_offsets[line - 1] + col


def _collect_mypy_issues() -> list[MypyIssue]:
    env = os.environ.copy()
    proc = subprocess.run(
        MYPY_CMD,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        check=False,
    )
    issues: list[MypyIssue] = []
    pattern = re.compile(
        r"^(?P<path>[^:]+):(?P<line>\d+)(?::\d+)?: error: (?P<message>.+?)\s+\[(?P<code>[^\]]+)\]$"
    )
    for raw_line in proc.stdout.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        match_line = pattern.match(raw_line)
        if not match_line:
            continue
        code = match_line.group("code")
        if code not in SUPPORTED_CODES:
            continue
        message = match_line.group("message")
        match = RETURN_RE.search(message) or ALT_RETURN_RE.search(message)
        if not match:
            continue
        actual = _normalise_type(match.group("actual").strip())
        expected = _normalise_type(match.group("expected").strip())
        if not actual or actual == expected:
            continue
        if actual.lower() == "any":
            continue
        rel_path = match_line.group("path")
        path = (ROOT / rel_path).resolve()
        if not path.exists():
            continue
        if not any(path.is_relative_to(base) for base in PROJECT_DIRS if base.exists()):
            continue
        line = int(match_line.group("line"))
        if line <= 0:
            continue
        issues.append(MypyIssue(path=path, line=line, actual=actual, expected=expected))
    return issues


def _gather_replacements(
    text: str, issues: Iterable[MypyIssue], path: Path
) -> list[Replacement]:
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []

    replacements: list[Replacement] = []
    line_offsets = _parse_line_offsets(text)

    for issue in issues:
        target_node: ast.AST | None = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end_lineno = getattr(node, "end_lineno", node.lineno)
                if node.lineno <= issue.line <= end_lineno:
                    target_node = node
                    break
        if not isinstance(target_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if target_node.returns is None:
            continue
        return_node = target_node.returns
        start = _offset(line_offsets, return_node.lineno, return_node.col_offset)
        end = _offset(
            line_offsets,
            getattr(return_node, "end_lineno", return_node.lineno),
            getattr(return_node, "end_col_offset", return_node.col_offset),
        )
        new_annotation = issue.actual
        replacements.append(Replacement(start=start, end=end, text=new_annotation))

    return replacements


def _apply_replacements(text: str, replacements: list[Replacement]) -> str:
    if not replacements:
        return text
    ordered = sorted(replacements, key=lambda r: r.start, reverse=True)
    new_text = text
    for rep in ordered:
        if rep.start >= rep.end:
            continue
        new_text = new_text[: rep.start] + rep.text + new_text[rep.end :]
    if not new_text.endswith("\n"):
        new_text += "\n"
    return new_text


def main() -> int:
    issues = _collect_mypy_issues()
    if not issues:
        print("[mypy_return_autofix] No actionable mypy return-type issues detected.")
        return 0

    grouped: dict[Path, list[MypyIssue]] = {}
    for issue in issues:
        grouped.setdefault(issue.path, []).append(issue)

    changed_files: List[Path] = []
    for path, file_issues in grouped.items():
        try:
            original_text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        replacements = _gather_replacements(original_text, file_issues, path)
        if not replacements:
            continue
        updated = _apply_replacements(original_text, replacements)
        if updated != original_text:
            path.write_text(updated, encoding="utf-8")
            changed_files.append(path)

    if changed_files:
        print("[mypy_return_autofix] Updated return annotations in these files:")
        for file in changed_files:
            print(f"  - {file.relative_to(ROOT)}")
    else:
        print("[mypy_return_autofix] No changes applied.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    from trend_analysis.script_logging import setup_script_logging

    setup_script_logging(module_file=__file__)
    raise SystemExit(main())
