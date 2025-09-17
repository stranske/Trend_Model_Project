#!/usr/bin/env python
"""Automatically apply lightweight type-hygiene adjustments.

Goals (automation-safe):
  * Install missing type stubs (handled by workflow separately) – this script only mutates source files.
  * Inject `# type: ignore[import-untyped]` comments for a curated allowlist of known untyped imports
    (currently only `yaml`).
  * Idempotent: never duplicate ignore comments.
  * Skip legacy / excluded paths (Old/, notebooks/old/).
  * Avoid masking *semantic* errors – we only touch import lines that succeed at runtime but lack stubs.

Non-goals:
  * Do not auto-ignore undefined names, return type mismatches, or attr errors.
  * Do not blanket `type: ignore` whole files.

Invocation: safe to run multiple times; exits 0 even if no changes.

Config via environment variables:
  AUTO_TYPE_ALLOWLIST   Comma-separated module base names to treat as untyped (default: "yaml")
  AUTO_TYPE_DRY_RUN     If set to '1' print planned changes but do not write.

This script intentionally uses only the standard library.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIRS = [ROOT / "src", ROOT / "tests"]
EXCLUDE_PATTERNS = [
    re.compile(r"(^|/)Old(/|$)"),
    re.compile(r"(^|/)notebooks/old(/|$)"),
]
ALLOWLIST = [
    m.strip()
    for m in os.environ.get("AUTO_TYPE_ALLOWLIST", "yaml").split(",")
    if m.strip()
]
DRY_RUN = os.environ.get("AUTO_TYPE_DRY_RUN") == "1"

IMPORT_PATTERN = re.compile(
    r"^(?P<indent>\s*)(import|from)\s+(?P<module>[a-zA-Z0-9_\.]+)(?P<rest>.*)$"
)
IGNORE_TOKEN = "# type: ignore[import-untyped]"


def should_exclude(path: Path) -> bool:
    rel = str(path.as_posix())
    return any(p.search(rel) for p in EXCLUDE_PATTERNS)


def needs_ignore(module: str) -> bool:
    base = module.split(".")[0]
    return base in ALLOWLIST


def process_file(path: Path) -> tuple[bool, list[str]]:
    """Return (changed, new_lines)."""
    try:
        original = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return False, []

    changed = False
    new_lines: list[str] = []

    for line in original:
        m = IMPORT_PATTERN.match(line)
        if not m:
            new_lines.append(line)
            continue
        module = m.group("module")
        if needs_ignore(module) and IGNORE_TOKEN not in line:
            # Do not append if another type: ignore already covers it more specifically
            if "type: ignore" not in line:
                line = f"{line.rstrip()}  {IGNORE_TOKEN}"
                changed = True
        new_lines.append(line)

    return changed, new_lines


def iter_python_files():
    for base in SRC_DIRS:
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if should_exclude(path):
                continue
            yield path


def main() -> int:
    modified = []
    for py_file in iter_python_files():
        changed, new_lines = process_file(py_file)
        if changed:
            modified.append(py_file)
            if not DRY_RUN:
                py_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    if modified:
        rels = [str(p.relative_to(ROOT)) for p in modified]
        print(
            f"[auto_type_hygiene] Added import-untyped ignores to {len(modified)} file(s):"
        )
        for r in rels:
            print(f"  - {r}")
    else:
        print("[auto_type_hygiene] No changes needed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
