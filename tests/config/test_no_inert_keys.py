"""Regression guard against inert keys in shipped YAML configs (A2 / #5400).

A key is *inert* when it is shipped in a config YAML source but no engine code
path under ``src/`` ever reads it, so setting it is a silent no-op. A2 removed the
specific inert keys enumerated in issue #5400; this test keeps them from creeping
back and makes any future inert key visible (it must be either wired to a consumer
or consciously added to the allowlist below with a justification).

The consumer search deliberately **excludes** ``schema_generator.py`` -- that module
only *declares* keys (human descriptions + JSON-schema constraints), it does not
*consume* them. Including it would make a description-only key look "consumed" and
defeat the very regression this guards against.

Two categories of leaf are exempt from the "must be consumed" rule:

* ``FREEFORM_SECTIONS`` -- mapping sections whose children are user-supplied data
  (benchmark labels, per-asset overrides, blended-weight metric names, ...). Their
  child keys are values, not schema keys, so requiring a literal ``src/`` consumer
  for each is meaningless.
* ``ALLOWLIST_PATHS`` -- keys that are declared config surface but not (yet) read by
  the engine, retained outside A2's enumerated deletion set. Each entry carries a
  justification so the allowlist stays auditable and small.
"""

from __future__ import annotations

import ast
import tokenize
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

from trend_analysis.config import schema_generator

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
SRC = REPO_ROOT / "src"
SCHEMA_GENERATOR = SRC / "trend_analysis" / "config" / "schema_generator.py"

# Mapping sections whose children are dynamic, user-supplied values rather than
# fixed schema keys. We do not descend into these, but we still require the
# section key itself to have a consumer.
FREEFORM_SECTIONS: frozenset[str] = frozenset(schema_generator._FREEFORM_MAPS)
DYNAMIC_VALUE_SECTIONS: frozenset[str] = frozenset(
    {
        # Asset-name -> group-name mapping. The section is consumed as a mapping;
        # its children are user-supplied asset labels, not fixed config keys.
        "portfolio.constraints.groups",
    }
)

# Declared config surface that no engine path currently reads, retained outside
# A2's (#5400) enumerated inert-key deletion set. Keyed by exact dotted path so a
# future *different* key sharing a leaf name is still caught.
ALLOWLIST_PATHS: dict[str, str] = {
    # Dead parallelism alias; consolidation onto the canonical run.jobs is owned by
    # A20 / #5401, so it is left in place rather than removed by A2.
    "run.n_jobs": "Parallelism-key consolidation owned by A20/#5401.",
    # Data-ingestion keys carried by shipped/validated configs and flagged in the
    # #5389 (A1) strict-config analysis as legitimate-but-unwired surface. Whether
    # to wire or forbid them is the open owner decision in A1, so A2 leaves them.
    "data.indices_glob": "Declared ingestion glob (sibling of consumed managers_glob); A1/#5389 owner decision.",
    "data.price_column": "Declared ingestion key; A1/#5389 strict-config owner decision pending.",
    "data.currency": "Declared ingestion key; A1/#5389 strict-config owner decision pending.",
    "data.lookback_required": "Declared ingestion key; A1/#5389 strict-config owner decision pending.",
    # Declared feature flags retained as config surface (not in A2's enumerated set).
    "preprocessing.winsorise.limits": "Declared winsorisation settings; not in A2's enumerated inert set.",
    "preprocessing.de_duplicate": "Declared preprocessing flag; not in A2's enumerated inert set.",
    "preprocessing.log_prices": "Declared preprocessing flag; not in A2's enumerated inert set.",
    "preprocessing.resample.business_only": "Declared resample flag; not in A2's enumerated inert set.",
    "sample_split.rolling_walk": "Declared future-phase feature flag (see inline comment in defaults.yml).",
    "export.include_figures": "Declared Phase-2 export flag (see inline comment in defaults.yml).",
    "multi_period.weight_curve.anchors": "Declared multi-period weighting surface; not in A2's enumerated inert set.",
    "portfolio.rank.limit_one_per_firm": "Legacy universe-config ranking toggle; wiring/removal is outside A2/#5400.",
}


def _leaf_paths(node: Any, prefix: str = "") -> Iterator[tuple[str, str]]:
    """Yield ``(dotted_path, leaf_name)`` for every leaf in a parsed YAML mapping.

    A *leaf* is any value that is not a non-empty mapping: scalars, lists, ``None``,
    and empty mappings (``{}``) are all leaves. Lists are not descended into. Free-form
    mapping sections are not descended into either.
    """

    if prefix in FREEFORM_SECTIONS or prefix in DYNAMIC_VALUE_SECTIONS:
        yield prefix, prefix.rsplit(".", 1)[-1]
        return
    if isinstance(node, dict) and node:
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _leaf_paths(value, path)
    elif prefix:
        yield prefix, prefix.rsplit(".", 1)[-1]


def _src_string_literals() -> set[str]:
    literals: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        if path == SCHEMA_GENERATOR:
            continue
        with path.open("rb") as handle:
            for token in tokenize.tokenize(handle.readline):
                if token.type != tokenize.STRING:
                    continue
                try:
                    value = ast.literal_eval(token.string)
                except (SyntaxError, ValueError):
                    continue
                if isinstance(value, str):
                    literals.add(value)
    return literals


def _config_leaf_paths(paths: Iterable[Path]) -> list[tuple[Path, str, str]]:
    leaves: list[tuple[Path, str, str]] = []
    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict) and data, f"{path.relative_to(REPO_ROOT)} did not parse to a mapping"
        leaves.extend((path, dotted, leaf) for dotted, leaf in _leaf_paths(data))
    return leaves


def test_every_shipped_config_key_is_consumed() -> None:
    config_paths = schema_generator.collect_config_sources(CONFIG_DIR)
    leaves = _config_leaf_paths(config_paths)
    assert leaves, "expected to collect leaf keys from shipped config YAML sources"

    string_literals = _src_string_literals()
    inert: list[str] = []
    for path, dotted_path, leaf in leaves:
        if dotted_path in ALLOWLIST_PATHS:
            continue
        if leaf in string_literals or dotted_path in string_literals:
            continue
        inert.append(f"{path.relative_to(REPO_ROOT)}:{dotted_path}")

    assert not inert, (
        "Inert config keys declared in shipped YAML configs with no src/ consumer "
        "(neither matched as an exact Python string literal outside schema_generator.py "
        "nor listed in ALLOWLIST_PATHS). Wire a real consumer "
        f"or, if intentionally unwired, add a justified allowlist entry: {inert}"
    )
