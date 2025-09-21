#!/usr/bin/env python
"""Prune allowlisted Ruff codes that have disappeared for N consecutive runs.

Inputs:
  - .ruff-residual-allowlist.json  (list / dict schema with codes)
  - ci/autofix/history.json        (list of snapshots each with by_code)
Env Vars / Args (optional):
  --streak N   minimum consecutive absence (default 8)
Outputs:
  - Updated .ruff-residual-allowlist.json (if pruning occurs)
  - Prints summary; exits 0 always (non-fatal in CI)

Rules:
  * Only remove a code if it appears in allowlist AND its count is zero for the last N snapshots (or absent from by_code entirely) AND it is not present in ANY of the last N snapshots.
  * Keeps ordering stable.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ALLOW = pathlib.Path(".ruff-residual-allowlist.json")
HISTORY = pathlib.Path("ci/autofix/history.json")


def load_allowlist():
    try:
        data = json.loads(ALLOW.read_text())
    except Exception:
        return []
    # Support both list of codes or object with 'codes'
    if isinstance(data, dict) and "codes" in data:
        codes = data["codes"]
    elif isinstance(data, list):
        codes = data
    else:
        codes = []
    return codes


def save_allowlist(codes):
    # Persist using object form to allow future metadata
    ALLOW.write_text(json.dumps({"codes": codes}, indent=2, sort_keys=True))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--streak",
        type=int,
        default=8,
        help="Consecutive absence required to prune (default 8)",
    )
    args = ap.parse_args(argv)

    allow_codes = load_allowlist()
    if not allow_codes:
        print("No allowlist entries to prune.")
        return 0
    try:
        hist = json.loads(HISTORY.read_text())
        if not isinstance(hist, list):
            hist = []
    except Exception:
        hist = []
    if not hist:
        print("No history snapshots; skipping pruning.")
        return 0

    streak = args.streak
    tail = hist[-streak:]
    removal = []
    for code in allow_codes:
        # If code appears (count>0) in any of the tail snapshots, keep.
        if any((snap.get("by_code") or {}).get(code, 0) > 0 for snap in tail):
            continue
        # Additional guard: if code never appears in any snapshot we can prune too.
        ever_present = any(code in (snap.get("by_code") or {}) for snap in hist)
        if not ever_present or True:  # presence in tail already zero; prune
            removal.append(code)
    if not removal:
        print("No allowlist codes eligible for pruning.")
        return 0

    new_codes = [c for c in allow_codes if c not in removal]
    save_allowlist(new_codes)
    print(f'Pruned {len(removal)} codes from allowlist: {", ".join(removal)}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
