#!/usr/bin/env python
"""Generate residual Ruff markdown report from classification JSON.
Writes: ci/autofix/residual_report.md
"""
from __future__ import annotations

import datetime
import json
import pathlib

from trend_analysis.script_logging import setup_script_logging

setup_script_logging(module_file=__file__, announce=False)

root = pathlib.Path("ci/autofix")
root.mkdir(parents=True, exist_ok=True)
cls = {}
try:
    cls = json.loads((root / "ruff_classification.json").read_text())
except Exception:
    pass
lines = [
    "# Residual Ruff Report",
    "",
    f"Generated: {datetime.datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC",
    "",
    f"Total diagnostics: {cls.get('total', 0)}",
    f"New diagnostics: {cls.get('new', 0)}",
    f"Allowed diagnostics: {cls.get('allowed', 0)}",
]

# If trend file exists, show overall sparklines and top code trends
trend_path = root / "trend.json"
try:
    trend = json.loads(trend_path.read_text())
except Exception:
    trend = {}
if trend:
    lines += [
        "",
        "## Trend (last 40 runs)",
        "",
        f"Remaining: {trend.get('remaining_latest', 0)}  {trend.get('remaining_spark', '')}",
        f"New: {trend.get('new_latest', 0)}  {trend.get('new_spark', '')}",
    ]
    codes = trend.get("codes") or {}
    if codes:
        lines += ["", "### Top Codes Trend"]
        for code, data in codes.items():
            lines.append(f"- {code}: {data.get('latest', 0)}  {data.get('spark', '')}")

lines += ["", "## Per-Code Breakdown"]
by_code = cls.get("by_code", {})
for code, count in sorted(by_code.items(), key=lambda x: (-x[1], x[0])):
    lines.append(f"- {code}: {count}")
(root / "residual_report.md").write_text("\n".join(lines))
print("Wrote ci/autofix/residual_report.md")
