#!/usr/bin/env python
"""Generate ASCII sparklines for residual Ruff diagnostics trend.

Reads: ci/autofix/history.json
Writes:
  ci/autofix/trend.json (latest metrics + sparkline strings)
Prints a one-line summary for workflow consumption.

Sparkline chars chosen for good monotonic density distribution.
"""
from __future__ import annotations

import collections
import json
import pathlib

HISTORY = pathlib.Path("ci/autofix/history.json")
OUT = pathlib.Path("ci/autofix/trend.json")
SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(series: list[int]) -> str:
    if not series:
        return ""
    mn = min(series)
    mx = max(series)
    if mx == mn:
        return SPARK_CHARS[0] * len(series)
    span = mx - mn
    out = []
    for v in series:
        idx = int((v - mn) / span * (len(SPARK_CHARS) - 1))
        out.append(SPARK_CHARS[idx])
    return "".join(out)


def main() -> int:
    try:
        hist = json.loads(HISTORY.read_text())
        if not isinstance(hist, list):
            hist = []
    except Exception:
        hist = []
    remaining_series = [h.get("remaining", 0) for h in hist][-40:]  # last 40 points
    new_series = [h.get("new", 0) for h in hist][-40:]

    # Build per-code time series (last 40) for top residual codes ranked by latest count
    code_counts_latest = collections.Counter()
    for snap in hist[-1:]:  # only latest snapshot for ranking
        for code, c in (snap.get("by_code") or {}).items():
            if isinstance(c, int):
                code_counts_latest[code] += c
    top_codes = [code for code, _ in code_counts_latest.most_common(6)]  # limit to 6
    code_series = {code: [] for code in top_codes}
    for snap in hist[-40:]:
        by_code = snap.get("by_code") or {}
        for code in top_codes:
            code_series[code].append(int(by_code.get(code, 0)))

    code_sparklines = {
        code: {"latest": series[-1] if series else 0, "spark": sparkline(series)}
        for code, series in code_series.items()
    }
    trend = {
        "points": len(hist),
        "remaining_latest": remaining_series[-1] if remaining_series else 0,
        "new_latest": new_series[-1] if new_series else 0,
        "remaining_spark": sparkline(remaining_series),
        "new_spark": sparkline(new_series),
        "codes": code_sparklines,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(trend, indent=2, sort_keys=True))
    print(
        f"trend remaining={trend['remaining_latest']} new={trend['new_latest']} {trend['remaining_spark']} / {trend['new_spark']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
