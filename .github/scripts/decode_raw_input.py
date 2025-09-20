#!/usr/bin/env python3
"""Decode JSON-encoded raw_input passed via workflow dispatch and write input.txt

Reads raw_input.json (single JSON string) -> writes decoded text to input.txt if non-empty.
Falls back to treating file contents as plain text if JSON parse fails.
"""
from __future__ import annotations
import json
import re
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

    # Heuristic: if the input lost original line breaks (appears mostly as one very long line)
    # reconstruct newlines before common enumeration patterns so the parser can split topics.
    if text and ("\n" not in text or text.count("\n") < 2):
        # Insert a newline before occurrences of pattern like ' 2)' ' 3.' ' 4:' etc. when preceded by whitespace
        # and a digit/letter enumerator. We avoid touching inside code blocks by the simplicity of early stage.
        pattern = re.compile(
            r"(?:(?<=\s)|^)(?P<enum>([0-9]{1,3}|[A-Za-z]|[A-Za-z][0-9]+)[\)\.:\-])\s+"
        )
        # To keep it simple, split on matches and rejoin with newline+token.
        parts = []
        last = 0
        for m in pattern.finditer(text):
            start = m.start()
            if start > last:
                segment = text[last:start]
                parts.append(segment)
            # Start a new line at enumerator
            parts.append("\n" + text[m.start() : m.end()])
            last = m.end()
        if last < len(text):
            parts.append(text[last:])
        rebuilt = "".join(parts)
        # If heuristic produced more newlines (improves structure), adopt it.
        if rebuilt.count("\n") > text.count("\n"):
            text = rebuilt.lstrip("\n")

    if text.strip():
        OUT_FILE.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
