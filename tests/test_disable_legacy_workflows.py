from __future__ import annotations

from pathlib import Path

from tools.disable_legacy_workflows import (
    CANONICAL_WORKFLOW_FILES,
    CANONICAL_WORKFLOW_NAMES,
    _extract_next_link,
    _normalize_allowlist,
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


def test_extract_next_link_handles_missing_header() -> None:
    assert _extract_next_link(None) is None
    assert _extract_next_link("") is None


def test_extract_next_link_scans_all_params_for_next_relation() -> None:
    header = '<https://api.github.com?page=2>; foo="bar"; rel="next"'
    assert _extract_next_link(header) == "https://api.github.com?page=2"


def test_extract_next_link_ignores_non_next_relations() -> None:
    header = ", ".join(
        [
            '<https://api.github.com?page=2>; rel="prev"',
            '<https://api.github.com?page=3>; rel="last"',
        ]
    )
    assert _extract_next_link(header) is None


def test_extract_next_link_handles_multiple_segments() -> None:
    header = ", ".join(
        [
            '<https://api.github.com?page=2>; rel="prev"',
            '<https://api.github.com?page=3>; type="json"; rel="next"',
        ]
    )
    assert _extract_next_link(header) == "https://api.github.com?page=3"


def test_normalize_allowlist_trims_and_splits_values() -> None:
    values = [" foo , bar", "baz", "", "bar"]
    assert _normalize_allowlist(values) == {"foo", "bar", "baz"}


def test_normalize_allowlist_skips_empty_tokens() -> None:
    assert _normalize_allowlist([" , , ", " "]) == set()
