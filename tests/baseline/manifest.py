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
from pathlib import Path
from typing import Any

from baseline_kit import CoverageManifest

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


def catalog_touched_keys(
    catalog: Mapping[str, Any],
    schema_leaves: set[str] | None = None,
) -> set[str]:
    """Every parameter key referenced by any scenario or toggle.

    A scenario may patch an *object-valued* config key -- e.g. clearing
    ``portfolio.custom_weights`` to null so ``portfolio.weighting_scheme`` can
    actually drive the allocation (see catalog ``weighting_risk_parity`` /
    issue #5537). The schema enumerates such objects only at their leaves
    (``portfolio.custom_weights.Mgr_01`` ...), so when ``schema_leaves`` is
    given, an object-parent key is expanded to its leaf children. That keeps
    coverage credit on the real leaves and avoids a false "unknown key" flag,
    while still surfacing genuine typos (a key that is neither a leaf nor the
    parent of one is left untouched so the schema guard still catches it).
    """
    raw: set[str] = set()
    for scen in catalog.get("scenarios", []) or []:
        for block in ("base", "control", "vary"):
            raw.update((scen.get(block) or {}).keys())
        if scen.get("param"):
            raw.add(scen["param"])
    for tog in catalog.get("toggles", []) or []:
        if tog.get("flag"):
            raw.add(tog["flag"])

    if not schema_leaves:
        return raw

    keys: set[str] = set()
    for key in raw:
        if key in schema_leaves:
            keys.add(key)
            continue
        children = {leaf for leaf in schema_leaves if leaf.startswith(f"{key}.")}
        keys.update(children or {key})
    return keys


def build_manifest(
    catalog: Mapping[str, Any], read_keys: set[str] | None = None
) -> CoverageManifest:
    """Build the generic coverage manifest from TMP's schema + catalog.

    The schema-walk (``schema_leaf_keys``) and catalog-key extraction
    (``catalog_touched_keys``) are TMP-specific; the manifest itself is the
    shared ``baseline_kit.CoverageManifest``.
    """
    leaves = schema_leaf_keys()
    return CoverageManifest(
        all_keys=leaves,
        touched_keys=catalog_touched_keys(catalog, leaves),
        priority_params=list(catalog.get("priority_params", []) or []),
        read_keys=set(read_keys or set()),
    )
