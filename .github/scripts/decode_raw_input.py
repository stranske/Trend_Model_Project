#!/usr/bin/env python3
"""Decode JSON-encoded raw_input passed via workflow dispatch and write input.txt

Reads raw_input.json (single JSON string) -> writes decoded text to input.txt if non-empty.
Falls back to treating file contents as plain text if JSON parse fails.
"""
from __future__ import annotations

import json
from pathlib import Path

RAW_FILE = Path("raw_input.json")
OUT_FILE = Path("input.txt")


def main() -> None:
    if not RAW_FILE.exists():
        return
    raw = RAW_FILE.read_text(encoding="utf-8")
    text: str = ""
    try:
        if raw not in ("", "null"):
            text = json.loads(raw)
        else:
            text = ""
    except Exception:
        # Treat as already unescaped plain text
        text = raw
    text = (text or "").rstrip("\r\n")
    if text.strip():
        OUT_FILE.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
