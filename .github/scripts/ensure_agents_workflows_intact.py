#!/usr/bin/env python3
"""Fail the build if critical agents workflows are deleted or renamed."""
from __future__ import annotations

import sys
from typing import Iterable

CRITICAL_FILES = {
    ".github/workflows/agents-63-chatgpt-issue-sync.yml",
    ".github/workflows/agents-63-codex-issue-bridge.yml",
    ".github/workflows/agents-70-orchestrator.yml",
}


def parse_diff(diff_lines: Iterable[str]) -> list[str]:
    """Return violation messages for rename or deletion attempts."""
    violations: list[str] = []

    for raw in diff_lines:
        line = raw.strip()
        if not line:
            continue

        parts = line.split("\t")
        status = parts[0]

        if status.startswith("R"):
            # Renames include a similarity score, e.g. R100\told\tnew
            if len(parts) < 3:
                violations.append(
                    f"Unable to parse rename entry: '{line}'. Treating as violation."
                )
                continue
            old_path, new_path = parts[1], parts[2]
            if old_path in CRITICAL_FILES or new_path in CRITICAL_FILES:
                violations.append(
                    f"Renaming protected workflow '{old_path}' → '{new_path}' requires maintainer override."
                )
            continue

        if len(parts) < 2:
            violations.append(f"Unable to parse diff entry: '{line}'.")
            continue

        path = parts[1]
        if path not in CRITICAL_FILES:
            continue

        if status == "D":
            violations.append(
                f"Deletion of protected workflow '{path}' is blocked."
            )

    return violations


def main() -> int:
    violations = parse_diff(sys.stdin)

    if not violations:
        print("Critical agents workflows intact.")
        return 0

    print("::error::Blocked attempt to delete or rename protected workflows:")
    for message in violations:
        print(f"::error::{message}")

    print(
        "Refer to docs/AGENTS_POLICY.md for emergency override steps and repository "
        "ruleset guidance."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
