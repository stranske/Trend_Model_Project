#!/usr/bin/env python
"""Render mypy JSON diagnostics into a plain-text summary.

Usage: ``python scripts/render_mypy_summary.py <input_json> <output_report>``
where ``input_json`` is the newline-delimited JSON produced by ``mypy
--error-format=json``.  If the file is missing or empty the script renders a
success message.  The script never raises on malformed entries – entries that
cannot be decoded are ignored so automation continues gracefully.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable


def _load_entries(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    entries: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            entries.append(data)
    return entries


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if len(args) != 2:
        print(
            "Usage: python scripts/render_mypy_summary.py <input_json> <output_report>",
            file=sys.stderr,
        )
        return 1

    input_path = Path(args[0])
    output_path = Path(args[1])

    lines: list[str] = []
    for entry in _load_entries(input_path):
        path = entry.get("path") or entry.get("file") or "<unknown>"
        line = entry.get("line", 0)
        column = entry.get("column", 0)
        severity = entry.get("severity", "error")
        message = entry.get("message", "")
        lines.append(f"{path}:{line}:{column}: {severity}: {message}")

    if lines:
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        output_path.write_text("Success: no mypy issues detected.\n", encoding="utf-8")

    return 0


if __name__ == "__main__":  # pragma: no cover
    from trend_analysis.script_logging import setup_script_logging

    setup_script_logging(module_file=__file__)
    raise SystemExit(main())
