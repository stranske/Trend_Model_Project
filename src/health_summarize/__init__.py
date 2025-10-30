"""Utilities for summarising health guardrail workflows.

This lightweight fallback mirrors the functionality that the
CI helpers expect from the external ``health_summarize`` package.
It is intentionally small so that the coverage workflow can run in
environments where the original dependency is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

_CHECK_SIGNATURE = "Health 43 CI Signature Guard"
_CHECK_BRANCH = "Health 44 Gate Branch Protection"
_DOC_ANCHOR = "docs/ci/WORKFLOWS.md#ci-signature-guard-fixtures"


_BOOL_TRUE = {"1", "true", "t", "yes", "y", "on"}
_BOOL_FALSE = {"0", "false", "f", "no", "n", "off"}


def _read_bool(value: object) -> bool:
    """Coerce truthy CLI values into a boolean.

    ``None`` defaults to ``False`` so optional flags behave sensibly.
    Strings are matched case-insensitively.
    """

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in _BOOL_TRUE:
        return True
    if text in _BOOL_FALSE:
        return False
    # Non-standard values fall back to Python truthiness so that
    # callers can still pass unusual markers (e.g. environment vars).
    return bool(text)


def _load_json(path: Path) -> object | None:
    """Return the decoded JSON payload if the file exists."""

    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _escape_table(text: str) -> str:
    """Escape vertical bars so markdown tables render correctly."""

    return text.replace("|", "&#124;")


def _doc_url() -> str:
    """Return the documentation URL for guardrail fixtures."""

    repo = os.getenv("GITHUB_REPOSITORY", "stranske/Trend_Model_Project")
    server = os.getenv("GITHUB_SERVER_URL", "https://github.com")
    base_ref = os.getenv("GITHUB_BASE_REF")
    ref_name = os.getenv("GITHUB_REF_NAME", "main")
    event = os.getenv("GITHUB_EVENT_NAME", "push")

    if event == "pull_request" and base_ref:
        ref = base_ref
    else:
        ref = ref_name or "main"
    return f"{server}/{repo}/blob/{ref}/{_DOC_ANCHOR}"


def build_signature_hash(jobs: list[dict]) -> str:
    """Derive the deterministic hash used for signature fixtures."""

    parts: list[str] = []
    for job in jobs:
        name = str(job.get("name", "?"))
        step = str(job.get("step", "no-step"))
        stack = str(job.get("stack", "no-stack"))
        parts.append(f"{name}::{step}::{stack}")
    parts.sort()
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:12]


def _signature_row(jobs_path: Path, expected_path: Path) -> dict[str, str]:
    """Summarise the CI signature guard status."""

    jobs_data = _load_json(jobs_path)
    if not isinstance(jobs_data, list):
        return {
            "check": _CHECK_SIGNATURE,
            "conclusion": "failure",
            "status": "Fixture unreadable",
            "details": "Unable to parse signature fixture; see docs for refresh steps.",
        }

    expected_sig = ""
    if expected_path.exists():
        expected_sig = expected_path.read_text(encoding="utf-8").strip()

    signature = build_signature_hash(jobs_data)
    if expected_sig and signature != expected_sig:
        return {
            "check": _CHECK_SIGNATURE,
            "conclusion": "failure",
            "status": f"Hash drift detected ({signature})",
            "details": (
                f"Computed {signature}, expected {expected_sig}. "
                f"Refresh fixtures documented in {_doc_url()}."
            ),
        }

    status = (
        f"✅ Signature current ({signature})"
        if expected_sig
        else f"✅ Computed {signature}"
    )
    details = (
        "Signature matches expected fixture."
        if expected_sig
        else "Signature generated from workflow jobs."
    )
    return {
        "check": _CHECK_SIGNATURE,
        "conclusion": "success",
        "status": status,
        "details": details,
    }


def _extract_contexts(section: object) -> list[str]:
    """Normalise the contexts list from a snapshot section."""

    if isinstance(section, str):
        return [section] if section else []
    if isinstance(section, Iterable) and not isinstance(section, dict):
        result: list[str] = []
        for item in section:
            if item:
                result.append(str(item))
        return result
    if isinstance(section, dict):
        contexts = section.get("contexts")
        return _extract_contexts(contexts)
    return []


def _format_bool(value: object | None) -> str:
    if value is None:
        return "❔ Unknown"
    return "✅ True" if bool(value) else "❌ False"


def _select_previous_section(snapshot: dict) -> dict:
    """Choose the most recent previous snapshot section."""

    if not isinstance(snapshot, dict):
        return {}
    for key in ("after", "desired", "current"):
        section = snapshot.get(key)
        if isinstance(section, dict):
            return section
    return {}


def _format_require_up_to_date(snapshot: dict) -> str:
    current = snapshot.get("current", {}) if isinstance(snapshot, dict) else {}
    current_strict = current.get("strict") if isinstance(current, dict) else None
    previous_section = _select_previous_section(snapshot)
    previous_strict = (
        previous_section.get("strict") if isinstance(previous_section, dict) else None
    )

    current_fmt = _format_bool(current_strict)
    if previous_strict is None or previous_strict == current_strict:
        return current_fmt
    previous_fmt = _format_bool(previous_strict)
    return f"{previous_fmt} → {current_fmt}"


def _format_delta(current: dict, previous_snapshot: dict | None) -> str:
    if not previous_snapshot:
        return "No previous snapshot"

    current_section = current.get("current") if isinstance(current, dict) else None
    if not isinstance(current_section, dict):
        current_section = {}
    previous_section = _select_previous_section(previous_snapshot)

    current_contexts = set(_extract_contexts(current_section))
    previous_contexts = set(_extract_contexts(previous_section))

    additions = sorted(current_contexts - previous_contexts)
    removals = sorted(previous_contexts - current_contexts)

    parts: list[str] = [f"+{ctx}" for ctx in additions]
    parts.extend(f"-{ctx}" for ctx in removals)

    current_strict = (
        current_section.get("strict") if isinstance(current_section, dict) else None
    )
    previous_strict = (
        previous_section.get("strict") if isinstance(previous_section, dict) else None
    )
    if previous_strict != current_strict:
        parts.append(
            "Require up to date: "
            f"{_format_bool(previous_strict)} → {_format_bool(current_strict)}"
        )

    return ", ".join(parts) if parts else "No changes"


def _snapshot_missing_detail(section: str, has_token: bool) -> tuple[str, str]:
    if has_token:
        return f"⚠️ {section}: Snapshot missing", "warning"
    return f"ℹ️ {section}: Observer mode – snapshot missing", "warning"


def _snapshot_detail(
    section: str,
    snapshot: dict | None,
    previous_snapshot: dict | None,
    *,
    has_token: bool,
) -> tuple[str, str]:
    """Summarise a single branch protection snapshot."""

    if not snapshot:
        return _snapshot_missing_detail(section, has_token)

    if snapshot.get("error"):
        return f"❌ {section}: {snapshot['error']}", "failure"

    status_bits: list[str] = []
    severity = "success"
    if snapshot.get("changes_required"):
        status_bits.append("Changes required")
        severity = "warning"
    else:
        status_bits.append("No changes required")

    if snapshot.get("changes_applied"):
        status_bits.append("Changes applied")
        severity = "warning"

    if snapshot.get("strict_unknown"):
        status_bits.append("Strict status unknown")
        severity = "warning"

    if snapshot.get("require_strict"):
        status_bits.append(f"Require up to date {_format_require_up_to_date(snapshot)}")

    delta = _format_delta(snapshot, previous_snapshot)
    if delta != "No previous snapshot":
        status_bits.append(f"Δ {delta}")

    icon = "✅" if severity == "success" else "⚠️"
    detail = f"{icon} {section}: " + "; ".join(status_bits)
    return detail, severity


_SEVERITY_ORDER = {"failure": 3, "warning": 2, "success": 1, "info": 0}


def _combine_severity(values: Iterable[str]) -> str:
    highest = "success"
    highest_score = -1
    for value in values:
        score = _SEVERITY_ORDER.get(value, 0)
        if score > highest_score:
            highest = value
            highest_score = score
    return highest


def _branch_row(snapshot_dir: Path, has_token: bool) -> dict[str, str]:
    """Aggregate enforcement and verification snapshots."""

    snapshot_dir = snapshot_dir.resolve()
    previous_dir = snapshot_dir / "previous"

    sections = ["Enforcement", "Verification"]
    details: list[str] = []
    severities: list[str] = []
    snapshots_found = False

    for section in sections:
        current_path = snapshot_dir / f"{section.lower()}.json"
        previous_path = previous_dir / f"{section.lower()}.json"
        current_snapshot = _load_json(current_path)
        previous_snapshot = (
            _load_json(previous_path) if previous_path.exists() else None
        )
        if current_snapshot:
            snapshots_found = True
        detail, severity = _snapshot_detail(
            section, current_snapshot, previous_snapshot, has_token=has_token
        )
        details.append(detail)
        severities.append(severity)

    if not snapshots_found:
        details = ["Observer mode – branch protection snapshots not present."]
        severities = ["warning"]

    conclusion = _combine_severity(severities)
    status = "Branch protection status"
    details_text = "\n".join(details)

    return {
        "check": _CHECK_BRANCH,
        "conclusion": conclusion,
        "status": status,
        "details": details_text,
    }


def _write_json(target: Path, rows: list[dict[str, str]]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _write_summary(target: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Health guardrail summary",
        "",
        "| Check | Status | Details |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        details = row.get("details", "") or "–"
        lines.append(
            f"| {_escape_table(row.get('check', '–'))} | "
            f"{_escape_table(row.get('status', '–'))} | "
            f"{_escape_table(details)} |"
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class Args:
    signature_jobs: Path | None = None
    signature_expected: Path | None = None
    snapshot_dir: Path | None = None
    has_enforce_token: bool = False
    write_json: Path | None = None
    write_summary: Path | None = None


def _parse_args(argv: Sequence[str] | None) -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signature-jobs")
    parser.add_argument("--signature-expected")
    parser.add_argument("--snapshot-dir")
    parser.add_argument("--has-enforce-token")
    parser.add_argument("--write-json")
    parser.add_argument("--write-summary")
    ns = parser.parse_args(list(argv) if argv is not None else None)
    return Args(
        signature_jobs=Path(ns.signature_jobs) if ns.signature_jobs else None,
        signature_expected=(
            Path(ns.signature_expected) if ns.signature_expected else None
        ),
        snapshot_dir=Path(ns.snapshot_dir) if ns.snapshot_dir else None,
        has_enforce_token=_read_bool(ns.has_enforce_token),
        write_json=Path(ns.write_json) if ns.write_json else None,
        write_summary=Path(ns.write_summary) if ns.write_summary else None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    rows: list[dict[str, str]] = []

    if args.signature_jobs:
        expected_path = args.signature_expected or args.signature_jobs.with_suffix(
            ".expected"
        )
        rows.append(_signature_row(args.signature_jobs, expected_path))

    if args.snapshot_dir:
        rows.append(_branch_row(args.snapshot_dir, args.has_enforce_token))

    if args.write_json:
        _write_json(args.write_json, rows)
    if args.write_summary and rows:
        _write_summary(args.write_summary, rows)

    return 0


__all__ = [
    "Args",
    "_branch_row",
    "_doc_url",
    "_escape_table",
    "_extract_contexts",
    "_format_delta",
    "_format_require_up_to_date",
    "_load_json",
    "_read_bool",
    "_select_previous_section",
    "_signature_row",
    "_snapshot_detail",
    "_write_json",
    "_write_summary",
    "build_signature_hash",
    "main",
]
