from __future__ import annotations

from pathlib import Path

import yaml


WORKFLOW_DIR = Path(".github/workflows")
ARCHIVE_DIR = Path("Old/workflows")
SELFTEST_PATH = WORKFLOW_DIR / "reusable-99-selftest.yml"


def test_selftest_workflow_inventory() -> None:
    """Exactly one active self-test workflow should be present."""

    selftest_workflows = sorted(
        path.name for path in WORKFLOW_DIR.glob("*selftest*.yml")
    )
    assert selftest_workflows == [
        "reusable-99-selftest.yml"
    ], "Active self-test inventory drifted; expected only reusable-99-selftest.yml."


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


def test_selftest_matrix_and_aggregate_contract() -> None:
    assert SELFTEST_PATH.exists(), "reusable-99-selftest.yml is missing from .github/workflows/"

    data = yaml.safe_load(SELFTEST_PATH.read_text())
    jobs = data.get("jobs", {})

    scenario_job = jobs.get("scenario")
    assert scenario_job is not None, "Scenario job missing from reusable-99-selftest.yml"
    assert (
        scenario_job.get("uses") == "./.github/workflows/reusable-90-ci-python.yml"
    ), "Scenario job must invoke reusable-90-ci-python.yml via jobs.<id>.uses"

    matrix = (
        scenario_job.get("strategy", {})
        .get("matrix", {})
        .get("include", [])
    )
    scenario_names = [entry.get("name") for entry in matrix]
    expected_names = [
        "minimal",
        "metrics_only",
        "metrics_history",
        "classification_only",
        "coverage_delta",
        "full_soft_gate",
    ]
    assert scenario_names == expected_names, (
        "Self-test scenario matrix drifted; update verification docs/tests if intentional."
    )

    aggregate_job = jobs.get("aggregate")
    assert aggregate_job is not None, "Aggregate verification job missing"
    assert (
        aggregate_job.get("needs") == "scenario"
    ), "Aggregate job must depend on the scenario matrix"
    assert (
        aggregate_job.get("if") == "${{ always() }}"
    ), "Aggregate job should always run to summarise results"

    permissions = aggregate_job.get("permissions", {})
    assert permissions.get("actions") == "read", "Aggregate job must read workflow artifacts"
    assert permissions.get("contents") == "read", "Aggregate job must read repository contents"

    env = aggregate_job.get("env", {})
    aggregate_list = env.get("SCENARIO_LIST", "")
    env_names = [name.strip() for name in aggregate_list.split(",") if name.strip()]
    assert env_names == expected_names, (
        "Aggregate SCENARIO_LIST must stay aligned with the scenario matrix to keep summaries accurate."
    )

    outputs = aggregate_job.get("outputs", {})
    assert {"verification_table", "failures"}.issubset(outputs), (
        "Aggregate job should surface verification outputs for downstream consumers."
    )
