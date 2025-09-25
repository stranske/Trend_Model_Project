"""Validation helper for the label-agent workflow.

This script runs inside the "Label agent PRs" workflow after we fetch the
trusted configuration via sparse checkout.  It verifies that the checkout only
contains allowlisted files so the workflow never evaluates untrusted
content.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable


def _parse_allowlist(raw: str) -> list[str]:
    entries: list[str] = []
    for line in raw.splitlines():
        trimmed = line.strip()
        if trimmed:
            entries.append(trimmed.replace("\\", "/"))
    return entries


def _iter_checkout_files(root: Path) -> Iterable[str]:
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part == ".git" for part in path.parts):
            continue
        yield path.relative_to(root).as_posix()


def main() -> int:
    raw_allowlist = os.environ.get("TRUSTED_LABEL_RULE_PATHS", "")
    allowlist = _parse_allowlist(raw_allowlist)
    if not allowlist:
        print(
            "[label-rules-assert] TRUSTED_LABEL_RULE_PATHS must contain at least one entry",
            file=sys.stderr,
        )
        return 1

    checkout_root = Path("trusted-config")
    if not checkout_root.exists():
        print(
            "[label-rules-assert] Expected checkout directory 'trusted-config' not found",
            file=sys.stderr,
        )
        return 1

    missing = [entry for entry in allowlist if not (checkout_root / entry).exists()]
    if missing:
        print(
            "[label-rules-assert] Missing allowlisted paths:\n  - "
            + "\n  - ".join(missing),
            file=sys.stderr,
        )
        return 1

    allowset = set(allowlist)
    extras = [rel for rel in _iter_checkout_files(checkout_root) if rel not in allowset]
    if extras:
        print(
            "[label-rules-assert] Unexpected files present after sparse checkout:\n  - "
            + "\n  - ".join(sorted(extras)),
            file=sys.stderr,
        )
        return 1

    print("[label-rules-assert] Sparse checkout contents verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
