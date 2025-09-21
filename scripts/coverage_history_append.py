#!/usr/bin/env python3
"""Append a single coverage trend record to an NDJSON history file.

Idempotent for identical run_id: if a record with the same run_id exists it will be replaced.
Usage environment variables (set by workflow):
  HISTORY_PATH: path to ndjson file (default coverage-trend-history.ndjson)
  RECORD_PATH: path to JSON record (default coverage-trend.json)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip corrupt lines but keep file usable
                continue
    return records


def main() -> int:
    history_path = Path(os.environ.get("HISTORY_PATH", "coverage-trend-history.ndjson"))
    record_path = Path(os.environ.get("RECORD_PATH", "coverage-trend.json"))
    if not record_path.exists():
        print(f"[history] record file missing: {record_path}", file=sys.stderr)
        return 0
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[history] failed to parse record: {e}", file=sys.stderr)
        return 0
    existing = load_existing(history_path)
    # Replace any existing entry with same run_id
    run_id = record.get("run_id")
    if run_id is not None:
        existing = [r for r in existing if r.get("run_id") != run_id]
    existing.append(record)

    # Sort by run_number if available
    def sort_key(r: dict):
        return r.get("run_number") or r.get("run_id") or 0

    existing.sort(key=sort_key)
    tmp = history_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for r in existing:
            out.write(json.dumps(r, sort_keys=True) + "\n")
    tmp.replace(history_path)
    print(f"[history] appended coverage record run_id={run_id} -> {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
