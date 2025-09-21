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


def _normalize_allow_entry(entry):
    code = entry.get("code") if isinstance(entry, dict) else None
    if not code:
        return None
    path = entry.get("path") if isinstance(entry, dict) else None
    if path is not None:
        path = str(path)
    return {"code": str(code), "path": path}


def load_allowlist():
    try:
        raw = json.loads(ALLOW.read_text())
    except Exception:
        return {"allow": []}, []

    template = {}
    entries = []

    if isinstance(raw, dict):
        template = dict(raw)
        allow_section = raw.get("allow")
        if isinstance(allow_section, list):
            for entry in allow_section:
                norm = _normalize_allow_entry(entry)
                if norm:
                    entries.append(norm)
        elif isinstance(raw.get("codes"), list):
            entries = [{"code": str(code), "path": None} for code in raw["codes"] if code]
            template.pop("codes", None)
        else:
            template = {"allow": []}
    elif isinstance(raw, list):
        entries = [{"code": str(code), "path": None} for code in raw if code]
    else:
        template = {"allow": []}

    if "allow" not in template:
        template["allow"] = entries.copy()

    return template, entries


def save_allowlist(template, entries):
    data = dict(template)
    data["allow"] = entries
    ALLOW.write_text(json.dumps(data, indent=2, sort_keys=True))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--streak",
        type=int,
        default=8,
        help="Consecutive absence required to prune (default 8)",
    )
    args = ap.parse_args(argv)

    template, allow_entries = load_allowlist()
    if not allow_entries:
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
    seen = set()
    codes_in_order = []
    for entry in allow_entries:
        code = entry["code"]
        if code not in seen:
            seen.add(code)
            codes_in_order.append(code)

    removal = []
    for code in codes_in_order:
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

    removal_set = set(removal)
    new_entries = [entry for entry in allow_entries if entry["code"] not in removal_set]
    save_allowlist(template, new_entries)
    print(f'Pruned {len(removal)} codes from allowlist: {", ".join(removal)}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
