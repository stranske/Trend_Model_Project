#!/usr/bin/env python
"""Build the consolidated autofix PR status comment.

Reads available artifacts:

* `autofix_report_enriched.json` (optional; defaults to enriched report from the
  autofix composite action)
* `ci/autofix/history.json` (optional)
* `ci/autofix/trend.json` (optional)

Outputs markdown with a stable HTML marker so the workflow can upsert the
comment in place. The script may be imported as a helper or executed via CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence, cast

MARKER = "<!-- autofix-status: DO NOT EDIT -->"


JsonMapping = Mapping[str, Any]
JsonLike = JsonMapping | Sequence[Any]


def load_json(path: pathlib.Path) -> object | None:
    """Read JSON from *path* if it exists, returning ``None`` on failure."""

    try:
        return cast(object, json.loads(path.read_text()))
    except Exception:
        return None


def coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def coerce_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def format_timestamp(raw: str | None) -> str:
    if not raw:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    try:
        normalized = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
        stamp = datetime.fromisoformat(normalized)
    except ValueError:
        return raw
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def format_spark(series: object | None) -> str:
    if isinstance(series, str) and series:
        return series
    return "∅"


def _ensure_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _top_code_lines(codes: JsonMapping | None) -> Sequence[str]:
    mapping = _ensure_mapping(codes)
    if not mapping:
        return ()
    lines: list[str] = ["", "### Top residual codes", ""]
    for code, payload_obj in sorted(mapping.items()):
        payload = _ensure_mapping(payload_obj)
        latest = coerce_int(payload.get("latest"))
        spark = format_spark(payload.get("spark"))
        lines.append(f"- `{code}` latest: **{latest}**  `{spark}`")
    return lines


def _snapshot_code_lines(snapshot: JsonMapping | None) -> Sequence[str]:
    mapping = _ensure_mapping(snapshot)
    if not mapping:
        return ()
    sortable: list[tuple[str, int]] = []
    for code, count in mapping.items():
        sortable.append((str(code), coerce_int(count)))
    sortable.sort(key=lambda item: (-item[1], item[0]))
    if not sortable:
        return ()
    lines = ["", "## Current per-code counts"]
    for code, count in sortable[:15]:
        lines.append(f"- `{code}`: {count}")
    return lines


def build_comment(
    *,
    report_path: pathlib.Path | None = None,
    history_path: pathlib.Path | None = None,
    trend_path: pathlib.Path | None = None,
    pr_number: str | None = None,
) -> str:
    """Construct the autofix status markdown comment."""

    report_obj = (
        (load_json(report_path) if report_path else None)
        or load_json(pathlib.Path("autofix_report_enriched.json"))
        or {}
    )
    history_obj = (load_json(history_path) if history_path else None) or load_json(
        pathlib.Path("ci/autofix/history.json")
    )
    trend_obj = (
        (load_json(trend_path) if trend_path else None)
        or load_json(pathlib.Path("ci/autofix/trend.json"))
        or {}
    )

    report = _ensure_mapping(report_obj)
    trend = _ensure_mapping(trend_obj)
    classification = _ensure_mapping(report.get("classification"))

    changed = coerce_bool(report.get("changed"))
    changed_text = "True" if changed else "False"
    remaining = coerce_int(classification.get("total"))
    new = coerce_int(classification.get("new"))
    allowed = coerce_int(classification.get("allowed"))

    history_points = 0
    if isinstance(history_obj, Mapping):
        history_points = len(history_obj)
    elif isinstance(history_obj, Sequence) and not isinstance(
        history_obj, (str, bytes)
    ):
        history_points = len(history_obj)

    remaining_latest = coerce_int(trend.get("remaining_latest"), remaining)
    new_latest = coerce_int(trend.get("new_latest"), new)
    remaining_spark = format_spark(trend.get("remaining_spark"))
    new_spark = format_spark(trend.get("new_spark"))

    timestamp = classification.get("timestamp") or report.get("timestamp")
    run_timestamp = format_timestamp(timestamp if isinstance(timestamp, str) else None)

    status_icon = "✅"
    status_suffix = ""
    if new_latest > 0:
        status_icon = "⚠️"
        status_suffix = " new diagnostics detected"
    elif remaining_latest > 0:
        status_icon = "⚠️"
        status_suffix = " residual diagnostics remain"
    elif changed:
        status_suffix = " autofix updates applied"
    status_value = f"{status_icon}{status_suffix}".strip()

    trigger_conclusion = os.environ.get("AUTOFIX_TRIGGER_CONCLUSION")
    trigger_class = os.environ.get("AUTOFIX_TRIGGER_CLASS")
    trigger_reason = os.environ.get("AUTOFIX_TRIGGER_REASON")
    trigger_head = os.environ.get("AUTOFIX_TRIGGER_PR_HEAD")
    skip_reason = os.environ.get("AUTOFIX_SKIP_REASON")

    mode_env = (os.environ.get("AUTOFIX_MODE") or "").strip().lower()
    clean_label_env = (os.environ.get("AUTOFIX_CLEAN_LABEL") or "").strip()
    if mode_env == "clean":
        mode_display = "Tests-only cosmetic"
        if clean_label_env:
            mode_display = f"{mode_display} (`{clean_label_env}`)"
    elif mode_env:
        mode_display = mode_env.replace("_", " ").title()
    else:
        mode_display = "Standard"

    metrics_rows = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Status | {status_value} |",
        f"| Mode | {mode_display} |",
        f"| Changed files | {changed_text} |",
        f"| Remaining issues | {remaining} |",
        f"| New issues | {new} |",
        f"| Allowed (legacy) | {allowed} |",
    ]
    if trigger_conclusion:
        trigger_display = trigger_conclusion
        if trigger_class and trigger_class != trigger_conclusion:
            trigger_display = f"{trigger_conclusion} ({trigger_class})"
        metrics_rows.append(f"| Trigger | {trigger_display} |")
    if skip_reason:
        metrics_rows.append(f"| Skip reason | {skip_reason} |")
    if history_points:
        metrics_rows.append(f"| History points | {history_points} |")

    trend_lines = [
        f"Remaining: **{remaining_latest}**  `{remaining_spark}`",
        f"New: **{new_latest}**  `{new_spark}`",
    ]

    codes_section = _top_code_lines(trend.get("codes"))
    snapshot_section = _snapshot_code_lines(classification.get("by_code"))

    artifacts: list[str] = []
    if pr_number:
        artifacts.append(f"- JSON report: `autofix-report-pr-{pr_number}`")
        if history_points:
            artifacts.append(f"- Residual history: `autofix-history-pr-{pr_number}`")
    if not artifacts:
        artifacts.append("- No additional artifacts published for this run.")

    result_block = os.environ.get("AUTOFIX_RESULT_BLOCK")
    result_section: list[str] = []
    if result_block:
        result_lines = result_block.splitlines()
        if result_lines:
            result_section = ["## Autofix result", "", *result_lines, ""]

    meta_segments: list[str] = []
    if trigger_conclusion:
        meta_segments.append(f"conclusion={trigger_conclusion}")
    if trigger_reason:
        meta_segments.append(f"reason={trigger_reason}")
    if trigger_head:
        meta_segments.append(f"head={trigger_head}")
    meta_line = (
        f"<!-- autofix-meta: {' '.join(meta_segments)} -->" if meta_segments else None
    )

    lines = [
        MARKER,
        "# Autofix Status",
        "",
        f"*Run:* {run_timestamp}",
        "",
        *metrics_rows,
        "",
        *result_section,
        "## Trend (last 40 runs)",
        "",
        *trend_lines,
        *codes_section,
        *snapshot_section,
        "",
        "## Artifacts & Reports",
        "",
        *artifacts,
        "",
        "_This comment auto-updates; do not edit manually._",
        MARKER,
        "",
    ]

    if meta_line:
        lines.insert(1, meta_line)

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=None, help="Output file")
    parser.add_argument(
        "--report",
        type=pathlib.Path,
        default=pathlib.Path("autofix_report_enriched.json"),
        help="Path to enriched autofix report JSON",
    )
    parser.add_argument(
        "--history",
        type=pathlib.Path,
        default=pathlib.Path("ci/autofix/history.json"),
        help="Path to residual history JSON",
    )
    parser.add_argument(
        "--trend",
        type=pathlib.Path,
        default=pathlib.Path("ci/autofix/trend.json"),
        help="Path to residual trend JSON",
    )
    parser.add_argument(
        "--pr-number",
        dest="pr_number",
        default="",
        help="Pull request number for artifact naming",
    )
    args = parser.parse_args(argv)

    comment = build_comment(
        report_path=args.report,
        history_path=args.history,
        trend_path=args.trend,
        pr_number=args.pr_number or None,
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(comment)
    else:
        sys.stdout.write(comment)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
