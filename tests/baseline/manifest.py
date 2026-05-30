"""Input-coverage manifest -- the one genuinely custom piece of the kit.

No off-the-shelf tool maps each *app input parameter* to "is it exercised by a
blessed scenario?". This module does exactly that for TMP:

  * enumerate every leaf parameter from ``config.schema.json`` (the canonical
    input space) as dotted keys;
  * mark which keys a catalog scenario/toggle touches ("scenario coverage");
  * optionally fold in which keys the engine actually *read* at runtime
    ("read coverage" -- a wiring signal from TMP's ConfigCoverageTracker);
  * emit a human-readable markdown report.

The report is what the weekly issue automation can later consume to raise
"untested input element" / "unread parameter" issues.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .harness import REPO_ROOT

SCHEMA_PATH = REPO_ROOT / "config.schema.json"


def schema_leaf_keys(schema_path: Path = SCHEMA_PATH) -> set[str]:
    """Return the set of dotted leaf-parameter paths defined in the schema."""
    with schema_path.open() as fh:
        schema = json.load(fh)
    leaves: set[str] = set()
    _walk(schema, "", leaves)
    return leaves


def _walk(node: Any, prefix: str, leaves: set[str]) -> None:
    if not isinstance(node, Mapping):
        return
    props = node.get("properties")
    if isinstance(props, Mapping):
        for name, sub in props.items():
            path = f"{prefix}.{name}" if prefix else name
            sub_props = sub.get("properties") if isinstance(sub, Mapping) else None
            if isinstance(sub_props, Mapping):
                _walk(sub, path, leaves)
            else:
                leaves.add(path)
        return
    # Wrapper schemas (allOf/anyOf/oneOf) -- descend without extending prefix.
    for combiner in ("allOf", "anyOf", "oneOf"):
        for sub in node.get(combiner, []) or []:
            _walk(sub, prefix, leaves)


def catalog_touched_keys(catalog: Mapping[str, Any]) -> set[str]:
    """Every parameter key referenced by any scenario or toggle."""
    keys: set[str] = set()
    for scen in catalog.get("scenarios", []) or []:
        for block in ("base", "control", "vary"):
            keys.update((scen.get(block) or {}).keys())
        if scen.get("param"):
            keys.add(scen["param"])
    for tog in catalog.get("toggles", []) or []:
        if tog.get("flag"):
            keys.add(tog["flag"])
    return keys


@dataclass
class CoverageManifest:
    schema_keys: set[str]
    touched_keys: set[str]
    read_keys: set[str]
    priority_params: list[str]

    @property
    def unknown_catalog_keys(self) -> set[str]:
        """Catalog keys that are NOT valid schema leaves (typo guard)."""
        return {k for k in self.touched_keys if k not in self.schema_keys}

    @property
    def priority_gaps(self) -> list[str]:
        """Priority params with no scenario/toggle exercising them."""
        return [p for p in self.priority_params if p not in self.touched_keys]

    @property
    def scenario_coverage_pct(self) -> float:
        if not self.schema_keys:
            return 0.0
        return 100.0 * len(self.touched_keys & self.schema_keys) / len(self.schema_keys)

    def to_markdown(self) -> str:
        lines = [
            "# Baseline coverage manifest",
            "",
            f"- Schema parameters: **{len(self.schema_keys)}**",
            f"- Exercised by a scenario/toggle: **{len(self.touched_keys & self.schema_keys)}** "
            f"({self.scenario_coverage_pct:.1f}%)",
            f"- Observed read at runtime (baseline): **{len(self.read_keys)}**",
            "",
            "## Priority parameters",
            "",
        ]
        for p in self.priority_params:
            mark = "x" if p in self.touched_keys else " "
            lines.append(f"- [{mark}] `{p}`")
        if self.priority_gaps:
            lines += ["", "## Priority gaps (no scenario yet)", ""]
            lines += [f"- `{p}`" for p in self.priority_gaps]
        if self.unknown_catalog_keys:
            lines += ["", "## Catalog keys not found in schema (check spelling)", ""]
            lines += [f"- `{k}`" for k in sorted(self.unknown_catalog_keys)]
        return "\n".join(lines) + "\n"


def build_manifest(
    catalog: Mapping[str, Any], read_keys: set[str] | None = None
) -> CoverageManifest:
    return CoverageManifest(
        schema_keys=schema_leaf_keys(),
        touched_keys=catalog_touched_keys(catalog),
        read_keys=set(read_keys or set()),
        priority_params=list(catalog.get("priority_params", []) or []),
    )
