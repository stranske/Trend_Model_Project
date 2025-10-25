#!/usr/bin/env python3
"""Compose PR summary comment with marker for deduplication."""

import os
from pathlib import Path


def main() -> None:
    """Create gate summary markdown file with anchor marker."""
    body = (os.environ.get("SUMMARY_BODY") or "").strip()
    pr_number = os.environ.get("PR_NUMBER", "").strip()
    head_sha = os.environ.get("HEAD_SHA", "").strip()

    if not body:
        raise SystemExit("Summary body missing; aborting comment composition.")

    head_token = head_sha[:12] if head_sha else ""
    parts = []
    if pr_number:
        parts.append(f"pr={pr_number}")
    if head_token:
        parts.append(f"head={head_token}")
    if not parts:
        parts.append("anchor")

    marker = f"<!-- gate-summary: {' '.join(parts)} -->"
    payload = "\n".join([marker, body, ""])

    Path("gate-summary.md").write_text(payload, encoding="utf-8")
    print(marker)


if __name__ == "__main__":
    main()
