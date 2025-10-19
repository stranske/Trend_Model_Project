from __future__ import annotations

from pathlib import Path

from tools.disable_legacy_workflows import (
    CANONICAL_WORKFLOW_FILES,
    CANONICAL_WORKFLOW_NAMES,
)
from tests.test_workflow_naming import EXPECTED_NAMES


def test_canonical_workflow_files_match_inventory() -> None:
    on_disk = {path.name for path in Path(".github/workflows").glob("*.yml")}
    assert (
        on_disk == CANONICAL_WORKFLOW_FILES
    ), "Canonical workflow file allowlist drifted; update tools/disable_legacy_workflows.py."


def test_canonical_workflow_names_match_expected_mapping() -> None:
    assert (
        set(EXPECTED_NAMES) == CANONICAL_WORKFLOW_FILES
    ), "Workflow naming expectations drifted; keep EXPECTED_NAMES in sync with the allowlist."
    assert CANONICAL_WORKFLOW_NAMES == set(
        EXPECTED_NAMES.values()
    ), "Workflow display-name allowlist drifted; synchronize EXPECTED_NAMES in tests/test_workflow_naming.py."
