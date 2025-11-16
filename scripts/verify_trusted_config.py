#!/usr/bin/env python3
"""Verify sparse checkout only includes trusted config files (Issue #1140).

Reads TRUSTED_CONFIG_PATHS from environment (newline separated) and ensures:
- Each listed path exists as a file in the checkout root provided to the workflow (trusted-config).
- No extra files beyond the allowlist are present.
Exits non‑zero on violation, emits ::error:: annotations for GitHub Actions.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    allowed = {
        p.strip()
        for p in os.environ.get("TRUSTED_CONFIG_PATHS", "").splitlines()
        if p.strip()
    }
    if not allowed:
        print("::error::No trusted config paths defined", file=sys.stderr)
        return 1
    root = Path("trusted-config")
    if not root.exists():
        print("::error::trusted-config checkout missing", file=sys.stderr)
        return 1
    found: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if "/.git/" in f"/{rel}/" or rel.startswith(".git/"):
            continue
        found.add(rel)
    missing = sorted(allowed - found)
    extra = sorted(found - allowed)
    if missing:
        print(
            "::error::Missing required trusted config file(s): " + ", ".join(missing),
            file=sys.stderr,
        )
    if extra:
        print(
            "::error::Unexpected file(s) in trusted config checkout: "
            + ", ".join(extra),
            file=sys.stderr,
        )
    if missing or extra:
        return 1
    print("[verify_trusted_config] OK")
    return 0


if __name__ == "__main__":
    from trend_analysis.script_logging import setup_script_logging

    setup_script_logging(module_file=__file__)
    raise SystemExit(main())
