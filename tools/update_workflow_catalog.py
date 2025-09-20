#!/usr/bin/env python3
"""Generate tools/workflow_catalog.json from .github/workflows.

Heuristics:
- name: top-level 'name:' field; if missing, derive from filename.
- archived: true if 'ARCHIVED' appears in name or comment block + on: {}
- category classification via simple keyword rules.

Categories (heuristic order):
  ci: name contains any of ["CI", "Docker", "CodeQL", "Benchmark", "perf"]
  remediation: name contains ["autofix", "fix", "remediation"]
  governance: ["auto-merge", "automerge", "approve", "label", "stale", "path", "dependency", "quarantine"]
  agents: ["agent", "codex", "copilot"]
  release: ["release"]
  security: ["codeql", "dependency"]
  other: fallback

Note: Keep logic lightweight; manual curation can adjust JSON after generation.
"""
from __future__ import annotations
import json
import re
import sys
import pathlib
import datetime
import yaml
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
WF_DIR = ROOT / ".github" / "workflows"
CATALOG = ROOT / "tools" / "workflow_catalog.json"

CATEGORY_RULES = [
    ("ci", re.compile(r"\b(CI|Docker|Benchmark|perf)\b", re.I)),
    ("remediation", re.compile(r"autofix|remediation|fix", re.I)),
    (
        "governance",
        re.compile(
            r"auto-merge|automerge|approve|label|stale|path|dependency|quarantine", re.I
        ),
    ),
    ("agents", re.compile(r"agent|codex|copilot", re.I)),
    ("release", re.compile(r"release", re.I)),
    ("security", re.compile(r"codeql|dependency", re.I)),
]


def classify(name: str) -> str:
    for cat, rx in CATEGORY_RULES:
        if rx.search(name):
            return cat
    return "other"


def is_archived(data: dict[str, Any], text: str, name: str) -> bool:
    if "on" in data and data["on"] == {}:
        return True
    if "ARCHIVED" in name.upper():
        return True
    if "ARCHIVED WORKFLOW" in text.upper():
        return True
    return False


def extract_triggers(data: dict[str, Any]) -> Any:
    on = data.get("on", {})
    if isinstance(on, list):
        return {k: True for k in on}
    if isinstance(on, dict):
        out = {}
        for k, v in on.items():
            if k == "push":
                if isinstance(v, dict) and "branches" in v:
                    out[k] = v["branches"]
                else:
                    out[k] = True
            elif k == "workflow_run":
                if isinstance(v, dict) and "workflows" in v:
                    out[k] = v["workflows"]
                else:
                    out[k] = True
            else:
                out[k] = True
        return out
    return {}


def main() -> int:
    workflows = []
    for path in sorted(WF_DIR.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        try:
            data = yaml.safe_load(text) or {}
        except Exception as e:  # pragma: no cover
            print(f"WARN: YAML parse failed for {path.name}: {e}", file=sys.stderr)
            data = {}
        name = data.get("name") or path.stem
        archived = is_archived(data, text, name)
        triggers = extract_triggers(data)
        workflows.append(
            {
                "file": path.name,
                "name": name,
                "category": classify(name),
                "archived": archived,
                "triggers": triggers,
                "replacement_for": None,
            }
        )
    catalog = {
        "_meta": {
            "generated": datetime.datetime.utcnow().isoformat() + "Z",
            "source": ".github/workflows",
            "script": "tools/update_workflow_catalog.py",
        },
        "workflows": workflows,
    }
    CATALOG.write_text(
        json.dumps(catalog, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"Updated {CATALOG} with {len(workflows)} entries.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
