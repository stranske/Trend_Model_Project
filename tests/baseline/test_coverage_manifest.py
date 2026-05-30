"""Input-coverage manifest test -- emits the report and guards catalog quality.

Hard assertions (fail the suite):
  * every catalog key is a real schema parameter (typo guard);
  * every declared priority parameter is exercised by a scenario/toggle.

Soft output: writes ``reports/baseline/coverage.md`` -- the artifact the weekly
issue automation can later read to raise "untested input element" issues.
"""

from __future__ import annotations

from pathlib import Path

from . import manifest
from .conftest import load_catalog
from .harness import REPO_ROOT

# Written under docs/reports/ so the weekly repo-review evaluator (which globs
# docs/reports/*.md) discovers it. Commit this file for the evaluator to see it.
REPORT_PATH = REPO_ROOT / "docs" / "reports" / "baseline-coverage.md"


def test_catalog_keys_exist_in_schema():
    m = manifest.build_manifest(load_catalog())
    assert not m.unknown_catalog_keys, (
        "Catalog references parameters not found in config.schema.json "
        f"(typos?): {sorted(m.unknown_catalog_keys)}"
    )


def test_priority_params_are_covered():
    m = manifest.build_manifest(load_catalog())
    assert not m.priority_gaps, (
        "Priority parameters with no scenario/toggle: " + ", ".join(m.priority_gaps)
    )


def test_emit_coverage_report(baseline_output):
    """Write the coverage manifest, folding in runtime read-coverage."""
    m = manifest.build_manifest(
        load_catalog(), read_keys=baseline_output.config_keys_read
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(m.to_markdown())
    assert REPORT_PATH.exists()
