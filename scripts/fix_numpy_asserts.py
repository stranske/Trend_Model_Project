#!/usr/bin/env python
"""Rewrite specific NumPy assertions for autofix workflows.

This helper focuses on the diagnostic tests introduced to exercise the
automation pipeline. It replaces ``assert fancy_array == [...]`` style checks
with ``assert fancy_array.tolist() == [...]`` so NumPy no longer raises the
``truth value of an array is ambiguous`` error. Narrow scope keeps the script
predictable while still demonstrating the automation pathway.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_ROOT = ROOT / "tests"
TARGET_FILES = {
    Path("tests/test_pipeline_warmup_autofix.py"),
    Path("tests/test_rank_selection_core_unit.py"),
    Path("tests/test_selector_weighting.py"),
    Path("tests/test_autofix_repo_regressions.py"),
}
ASSERT_PATTERN = re.compile(
    r"^(?P<indent>\s*)assert\s+(?P<lhs>[A-Za-z_][A-Za-z0-9_\.]+)\s*==\s*(?P<rhs>\[.*?\])\s*(?P<comment>#.*)?$"
)


def _should_skip(lhs: str) -> bool:
    lower = lhs.lower()
    if "array" not in lower:
        return True
    if lhs.endswith(".tolist()"):
        return True
    if lhs.endswith(".to_list()"):
        return True
    return False


def _rewrite_line(line: str) -> str:
    match = ASSERT_PATTERN.match(line)
    if not match:
        return line
    lhs = match.group("lhs")
    if _should_skip(lhs):
        return line
    indent = match.group("indent")
    rhs = match.group("rhs").rstrip()
    comment = match.group("comment") or ""
    rewritten = f"{indent}assert {lhs}.tolist() == {rhs}"
    if comment:
        rewritten += f" {comment.strip()}"
    return rewritten


def process_file(path: Path) -> bool:
    rel_path = path.relative_to(ROOT)
    if rel_path not in TARGET_FILES:
        return False
    try:
        original = path.read_text(encoding="utf-8")
    except OSError:
        return False
    updated_lines = []
    changed = False
    for line in original.splitlines(keepends=True):
        stripped_line = line[:-1] if line.endswith("\n") else line
        new_stripped = _rewrite_line(stripped_line)
        new_line = new_stripped + ("\n" if line.endswith("\n") else "")
        if new_line != line:
            changed = True
        updated_lines.append(new_line)
    if changed:
        path.write_text("".join(updated_lines), encoding="utf-8")
    return changed


def main() -> int:
    if not TEST_ROOT.exists():
        print("[fix_numpy_asserts] tests/ directory not found; skipping")
        return 0
    changed_any = False
    for path in TEST_ROOT.rglob("test_*.py"):
        if process_file(path):
            print(
                f"[fix_numpy_asserts] Rewrote NumPy assert in {path.relative_to(ROOT)}"
            )
            changed_any = True
    if not changed_any:
        print("[fix_numpy_asserts] No eligible asserts found.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    from trend_analysis.script_logging import setup_script_logging

    setup_script_logging(module_file=__file__)
    raise SystemExit(main())
