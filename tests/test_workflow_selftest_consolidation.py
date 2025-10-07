from __future__ import annotations

from pathlib import Path


WORKFLOW_DIR = Path(".github/workflows")
ARCHIVE_DIR = Path("Old/workflows")


def test_selftest_workflows_removed_from_active_runs() -> None:
    selftest_workflows = sorted(
        path.name for path in WORKFLOW_DIR.glob("*selftest*.yml")
    )
    assert (
        selftest_workflows == []
    ), "Self-test workflows should be archived under Old/workflows/"


def test_archived_selftest_inventory() -> None:
    assert ARCHIVE_DIR.exists(), "Old/workflows directory is missing"

    archived_workflows = sorted(
        path.name for path in ARCHIVE_DIR.glob("*selftest*.yml")
    )
    assert archived_workflows == [
        "maint-90-selftest.yml",
        "reusable-99-selftest.yml",
    ], (
        "Archived self-test workflows are missing or unexpected files are present. "
        "Expected maint-90-selftest.yml and reusable-99-selftest.yml."
    )
