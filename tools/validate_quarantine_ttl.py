from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


@dataclasses.dataclass
class QuarantineRecord:
    """Single quarantine entry parsed from ``tests/quarantine.yml``."""

    identifier: str
    expires: dt.date
    raw: Dict[str, Any]


@dataclasses.dataclass
class ValidationReport:
    """Structured outcome of the TTL validation."""

    total_entries: int
    expired: List[QuarantineRecord]
    invalid: List[str]

    @property
    def ok(self) -> bool:
        return not self.expired and not self.invalid


def _parse_date(raw_value: Any, *, entry_id: str) -> Tuple[Optional[dt.date], Optional[str]]:
    if raw_value is None:
        return None, f"Entry `{entry_id or '<missing id>'}` has no expires field."
    if isinstance(raw_value, dt.date):
        return raw_value, None
    if isinstance(raw_value, str):
        try:
            return dt.date.fromisoformat(raw_value), None
        except ValueError:
            return (
                None,
                f"Entry `{entry_id or '<missing id>'}` uses non-ISO expires `{raw_value}`.",
            )
    return (
        None,
        f"Entry `{entry_id or '<missing id>'}` uses unsupported expires value {raw_value!r}.",
    )


def load_records(path: Path) -> Tuple[List[QuarantineRecord], List[str]]:
    if not path.exists():
        return [], []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    records: List[QuarantineRecord] = []
    invalid: List[str] = []
    for entry in data.get("tests", []) or []:
        if not isinstance(entry, dict):
            invalid.append("Non-mapping entry in quarantine list.")
            continue
        identifier = str(entry.get("id") or "").strip()
        expires_raw = entry.get("expires")
        expires, error = _parse_date(expires_raw, entry_id=identifier)
        if error:
            invalid.append(error)
            continue
        if not identifier:
            invalid.append("Entry missing `id` field.")
            continue
        if not isinstance(expires, dt.date):
            invalid.append(f"Entry `{identifier}` missing valid expires date.")
            continue
        records.append(QuarantineRecord(identifier=identifier, expires=expires, raw=entry))
    return records, invalid


def evaluate_records(
    records: Iterable[QuarantineRecord],
    *,
    today: Optional[dt.date] = None,
    additional_invalid: Optional[Sequence[str]] = None,
) -> ValidationReport:
    today = today or dt.date.today()
    expired: List[QuarantineRecord] = []
    invalid: List[str] = list(additional_invalid or [])
    total = 0

    for record in records:
        total += 1
        if record.expires < today:
            expired.append(record)

    return ValidationReport(total_entries=total, expired=expired, invalid=invalid)


def build_summary(report: ValidationReport) -> str:
    lines = ["## Quarantine TTL validation", ""]
    lines.append(f"- Total entries scanned: {report.total_entries}")
    if report.ok:
        lines.append("- ✅ No expired quarantines detected.")
    else:
        if report.expired:
            lines.append("- ❌ Expired quarantines detected:")
            for record in report.expired:
                lines.append(f"  - `{record.identifier}` expired on {record.expires.isoformat()}")
        if report.invalid:
            lines.append("- ⚠️ Entries with invalid TTL data:")
            for identifier in report.invalid:
                lines.append(f"  - {identifier}")
    return "\n".join(lines)


def emit_json(report: ValidationReport) -> str:
    return json.dumps(
        {
            "total_entries": report.total_entries,
            "expired": [
                {"id": record.identifier, "expires": record.expires.isoformat()}
                for record in report.expired
            ],
            "invalid": list(report.invalid),
            "ok": report.ok,
        },
        indent=2,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate quarantine TTLs")
    parser.add_argument(
        "path",
        nargs="?",
        default="tests/quarantine.yml",
        help="Path to the quarantine definition file (default: tests/quarantine.yml)",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print validation summary as JSON in addition to human-readable text.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    path = Path(args.path)
    records, invalid_entries = load_records(path)
    report = evaluate_records(records, additional_invalid=invalid_entries)
    summary = build_summary(report)

    step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary_path:
        with open(step_summary_path, "a", encoding="utf-8") as handle:
            handle.write(summary + "\n")

    print(summary)
    if args.json_output:
        print(emit_json(report))

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
