#!/usr/bin/env python
"""CI Feature Assertion

Validates presence (or absence) of artifacts / files based on feature flags passed via env.
This is a lightweight post-step guard to ensure the reusable workflow wiring remains intact.

Environment Variables:
  EXPECT_METRICS, EXPECT_HISTORY, EXPECT_CLASSIFICATION, EXPECT_COV_DELTA (values 'true' / 'false')
  HISTORY_ARTIFACT_NAME (expected history file path)
Exit codes:
  0 on success
  1 if any expectation violated
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from trend_analysis.script_logging import setup_script_logging

setup_script_logging(module_file=__file__, announce=False)


def expect(flag: str) -> bool:
    return str(os.environ.get(flag, "false")).lower() == "true"


errors: list[str] = []

# Metrics artifact check
if expect("EXPECT_METRICS"):
    if not Path("ci-metrics.json").is_file():
        errors.append("Expected metrics artifact ci-metrics.json not found")
else:
    if Path("ci-metrics.json").is_file():
        errors.append("Metrics file present but EXPECT_METRICS=false")

# Coverage delta check
if expect("EXPECT_COV_DELTA"):
    if not Path("coverage-delta.json").is_file():
        errors.append("Expected coverage-delta.json not found")
else:
    if Path("coverage-delta.json").is_file():
        errors.append("coverage-delta.json present but EXPECT_COV_DELTA=false")

# History & classification share script; validate separately
history_file = os.environ.get("HISTORY_ARTIFACT_NAME", "metrics-history.ndjson")
if expect("EXPECT_HISTORY"):
    if not Path(history_file).is_file():
        errors.append(f"History file {history_file} missing (EXPECT_HISTORY=true)")
else:
    if Path(history_file).is_file():
        errors.append(f"History file {history_file} present but EXPECT_HISTORY=false")

if expect("EXPECT_CLASSIFICATION"):
    if not Path("classification.json").is_file():
        errors.append("classification.json missing (EXPECT_CLASSIFICATION=true)")
else:
    if Path("classification.json").is_file():
        errors.append("classification.json present but EXPECT_CLASSIFICATION=false")

if errors:
    print("CI Feature Assertion FAIL:")
    for e in errors:
        print(" -", e)
    sys.exit(1)
else:
    print("CI Feature Assertion PASS")
