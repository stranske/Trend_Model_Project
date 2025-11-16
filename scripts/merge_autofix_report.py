#!/usr/bin/env python3
"""Merge autofix report meta information into the enriched JSON payload."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="input_path",
        default="autofix_report_enriched.json",
        help="Path to the enriched autofix report JSON file.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default="autofix_report.json",
        help="Destination path for the merged JSON report.",
    )
    parser.add_argument(
        "--pr-number",
        dest="pr_number",
        default="",
        help="Pull request number to embed in the report metadata.",
    )
    parser.add_argument(
        "--timestamp",
        dest="timestamp",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        help="UTC timestamp to record in the report metadata.",
    )
    return parser.parse_args()


def load_enriched(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Enriched report not found: {path}") from exc
    except json.JSONDecodeError:
        # Fall back to empty payload if we cannot parse the enriched file.
        return {}


def write_report(
    payload: object, output_path: Path, pr_number: str, timestamp: str
) -> None:
    meta = {
        "pull_request": pr_number,
        "timestamp_utc": timestamp,
    }

    if isinstance(payload, dict):
        payload.update(meta)
        data = payload
    else:
        data = {"meta": meta, "raw": payload}

    output_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    try:
        payload = load_enriched(input_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    write_report(payload, output_path, args.pr_number, args.timestamp)
    return 0


if __name__ == "__main__":
    from trend_analysis.script_logging import setup_script_logging

    setup_script_logging(module_file=__file__)
    raise SystemExit(main())
