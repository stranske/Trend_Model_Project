#!/usr/bin/env python
"""Build consolidated PR comment for autofix run.

Reads available artifacts:
  - autofix_report_enriched.json (optional)
  - ruff_classification.json (optional)
  - ci/autofix/trend.json (optional)
Outputs:
  - Writes markdown to stdout (so caller can redirect) OR to file path provided via --out.
Includes a stable HTML marker comment so workflow can update in-place.
"""
from __future__ import annotations
import json, pathlib, argparse, sys, datetime

MARKER = "<!-- autofix-status: DO NOT EDIT -->"

p = argparse.ArgumentParser()
p.add_argument(
    "--out", type=pathlib.Path, default=None, help="Write to file instead of stdout"
)
args = p.parse_args()

root = pathlib.Path(".")

# Load helpers


def load_json(path: pathlib.Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


enriched = load_json(pathlib.Path("autofix_report_enriched.json"))
classification = load_json(pathlib.Path("ruff_classification.json"))
trend = load_json(pathlib.Path("ci/autofix/trend.json"))

changed = str(enriched.get("changed", "unknown"))
cls = classification or enriched.get("classification", {})
remaining = cls.get("total")
new_issues = cls.get("new")
allowed = cls.get("allowed")
by_code = cls.get("by_code") or {}

lines: list[str] = []
lines.append(MARKER)
lines.append("# Autofix Status")
lines.append("")
lines.append(f"*Run:* {datetime.datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC")
lines.append("")


def severity_icon(rem: int | None, new: int | None) -> str:
    if rem is None:
        return "❔"
    if rem == 0:
        return "✅"
    if new and new > 0:
        return "🆕"
    if rem <= 10:
        return "🟢"
    if rem <= 50:
        return "🟡"
    return "🔴"


sev = severity_icon(remaining, new_issues)

lines.append("| Metric | Value |")
lines.append("|--------|-------|")
lines.append(f"| Status | {sev} |")
lines.append(f"| Changed files | {changed} |")
lines.append(f"| Remaining issues | {remaining if remaining is not None else 'NA'} |")
lines.append(f"| New issues | {new_issues if new_issues is not None else 'NA'} |")
lines.append(f"| Allowed (legacy) | {allowed if allowed is not None else 'NA'} |")

# Trend section
if trend:
    lines.append("")
    lines.append("## Trend (last 40 runs)")
    lines.append("")
    rem_latest = trend.get("remaining_latest", "NA")
    new_latest = trend.get("new_latest", "NA")
    lines.append(f"Remaining: **{rem_latest}**  `{trend.get('remaining_spark','')}`")
    lines.append(f"New: **{new_latest}**  `{trend.get('new_spark','')}`")
    codes = trend.get("codes") or {}
    if codes:
        lines.append("")
        lines.append("### Top Codes")
        for code, data in codes.items():
            lines.append(f"- `{code}`: {data.get('latest',0)} `{data.get('spark','')}`")

# Top offenders by code (current snapshot)
if by_code:
    lines.append("")
    lines.append("## Current Per-Code Counts")
    top_sorted = sorted(by_code.items(), key=lambda x: (-x[1], x[0]))[:15]
    for code, count in top_sorted:
        lines.append(f"- `{code}`: {count}")

lines.append("")
lines.append("")
lines.append("## Artifacts & Reports")
pr_number = enriched.get("pull_request") or ""
if pr_number:
    lines.append(f"- JSON report artifact: `autofix-report-pr-{pr_number}`")
    lines.append(f"- Patch artifact (forks only): `autofix-patch-pr-{pr_number}`")
lines.append(
    "- Residual markdown report emitted on scheduled cleanup runs (see branch `ci/autofix-residual-cleanup`)."
)

lines.append("")
lines.append("_This comment auto-updates; do not edit manually._")
lines.append(MARKER)

text = "\n".join(lines).strip() + "\n"
if args.out:
    args.out.write_text(text)
else:
    sys.stdout.write(text)
