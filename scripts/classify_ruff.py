#!/usr/bin/env python
"""Classify Ruff diagnostics into allowlisted vs new.

Usage:
    python scripts/classify_ruff.py ruff_diagnostics.json .ruff-residual-allowlist.json output.json

Input formats:
  ruff_diagnostics.json: JSON list produced by `ruff check --output-format json .`
  .ruff-residual-allowlist.json: {"allow": [{"code": "F401", "path": "src/pkg/module.py" | null}]}
     - Each allow entry may specify just a code (applies to all paths) or a code+path pair.

Output schema (written to output.json):
{
  "total": int,
  "allowed": int,
  "new": int,
  "by_code": {code: count},
  "new_by_code": {code: count},
  "allowed_by_code": {code: count},
  "timestamp": iso8601,
  "allowlist_size": int,
  "sample_new": [ up to 10 diagnostic dicts ],
  "sample_allowed": [ up to 10 diagnostic dicts ]
}

Non-fatal: any parsing error yields an empty classification with zero counts.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _normalize_allow_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    code = entry.get("code")
    if not code:
        raise ValueError("Allowlist entry missing 'code'")
    path = entry.get("path")
    return {"code": str(code), "path": str(path) if path else None}


def classify(
    diagnostics: List[Dict[str, Any]], allow: List[Dict[str, Any]]
) -> Dict[str, Any]:
    # Build lookup sets
    codes_global = {a["code"] for a in allow if a["path"] is None}
    code_path_pairs = {(a["code"], a["path"]) for a in allow if a["path"] is not None}

    out: Dict[str, Any] = {
        "total": 0,
        "allowed": 0,
        "new": 0,
        "by_code": {},
        "new_by_code": {},
        "allowed_by_code": {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "allowlist_size": len(allow),
        "sample_new": [],
        "sample_allowed": [],
    }

    for d in diagnostics:
        code = d.get("code") or "?"
        filepath = d.get("filename") or d.get("file") or ""
        out["by_code"][code] = out["by_code"].get(code, 0) + 1
        out["total"] += 1
        allowed_hit = (code in codes_global) or ((code, filepath) in code_path_pairs)
        if allowed_hit:
            out["allowed"] += 1
            out["allowed_by_code"][code] = out["allowed_by_code"].get(code, 0) + 1
            if len(out["sample_allowed"]) < 10:
                out["sample_allowed"].append({"code": code, "file": filepath})
        else:
            out["new"] += 1
            out["new_by_code"][code] = out["new_by_code"].get(code, 0) + 1
            if len(out["sample_new"]) < 10:
                out["sample_new"].append({"code": code, "file": filepath})

    return out


def main(argv: List[str]) -> int:
    if len(argv) != 4:
        print(
            "Usage: classify_ruff.py <ruff_diagnostics.json> <allowlist.json> <output.json>",
            file=sys.stderr,
        )
        return 2
    diag_path = Path(argv[1])
    allow_path = Path(argv[2])
    out_path = Path(argv[3])

    diagnostics_raw = _load_json(diag_path) or []
    allow_raw = _load_json(allow_path) or {}
    allow_entries = []
    try:
        for entry in allow_raw.get("allow", []):
            try:
                allow_entries.append(_normalize_allow_entry(entry))
            except Exception:
                continue
    except Exception:
        allow_entries = []

    # Ruff JSON sometimes wraps diagnostics in {"message":..., "code":..., "filename":...}
    diagnostics: List[Dict[str, Any]] = []
    if isinstance(diagnostics_raw, list):
        for item in diagnostics_raw:
            if isinstance(item, dict):
                diagnostics.append(item)
    # else fallback empty

    result = classify(diagnostics, allow_entries)
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
    except Exception as exc:  # pragma: no cover
        print(f"[classify_ruff] Failed to write output: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    from trend_analysis.script_logging import setup_script_logging

    setup_script_logging(module_file=__file__)
    raise SystemExit(main(sys.argv))
