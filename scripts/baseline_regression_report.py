#!/usr/bin/env python
"""Emit a baseline-regression report for the weekly repo-review to act on.

Runs the golden-master suite against the committed baselines. If any golden has
DRIFTED, writes ``docs/reports/baseline-regression.md`` describing what changed
so the weekly repo-review evaluator (which globs ``docs/reports/``) surfaces it
to the round-1 reviewer as an "investigate drift" candidate. If the goldens are
clean, any stale regression report is removed.

This closes the weekly drift loop:
  weekly run -> golden drift -> regression report -> evaluator -> issue.

It deliberately does NOT re-bless anything (that would mask regressions) and
always exits 0 (the workflow decides whether to fail); the artifact is the point.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "docs" / "reports" / "baseline-regression.md"
GOLDEN_TESTS = "tests/baseline/test_golden.py"


def main() -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", GOLDEN_TESTS, "-n0", "-q", "--no-header"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONHASHSEED": "0", **_env()},
    )
    passed = proc.returncode == 0

    if passed:
        if REPORT.exists():
            REPORT.unlink()
            print("Goldens clean; removed stale baseline-regression.md")
        else:
            print("Goldens clean; no regression report.")
        return 0

    failed = [
        line.split("::", 1)[-1].split(" ")[0]
        for line in proc.stdout.splitlines()
        if line.startswith("FAILED")
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(_render(failed, proc.stdout))
    print(f"Baseline regression detected ({len(failed)} golden(s)); wrote {REPORT.name}")
    return 0


def _env() -> dict[str, str]:
    import os

    return dict(os.environ)


def _render(failed: list[str], pytest_output: str) -> str:
    bullets = "\n".join(f"- `{name}`" for name in failed) or "- (see details below)"
    tail = "\n".join(pytest_output.splitlines()[-40:])
    return f"""# Baseline regression detected

The weekly baseline run found that committed golden masters no longer match the
current output. This is **either an unintended regression (fix it) or an intended
change (re-bless the goldens in a PR)** — it needs human adjudication.

## Drifted baselines

{bullets}

## What to do

1. Reproduce locally: `PYTHONHASHSEED=0 pytest {GOLDEN_TESTS} -n0`.
2. Inspect the diff. If the change is unintended, fix the regression.
3. If the change is intended, re-bless: `pytest {GOLDEN_TESTS} --force-regen`, review, commit.

## pytest output (tail)

```
{tail}
```
"""


if __name__ == "__main__":
    raise SystemExit(main())
