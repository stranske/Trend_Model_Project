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


def test_selftest_triggers_are_manual_only() -> None:
    """Self-test workflows must only expose manual or scheduled triggers."""

    selftest_files = sorted(WORKFLOW_DIR.glob("*selftest*.yml"))
    assert selftest_files, "Expected at least one self-test workflow definition."

    disallowed_triggers = {"pull_request", "pull_request_target", "push"}
    required_manual_trigger = "workflow_dispatch"
    optional_triggers = {"schedule", "workflow_call"}
    allowed_triggers = {required_manual_trigger} | optional_triggers

    for workflow_file in selftest_files:
        data = yaml.safe_load(workflow_file.read_text()) or {}

        triggers_raw = data.get("on")
        if triggers_raw is None and True in data:
            # `on:` is a YAML keyword; in 1.1 it can be parsed as boolean True.
            triggers_raw = data[True]
        if triggers_raw is None:
            triggers_raw = {}

        if isinstance(triggers_raw, list):
            triggers = {str(event): {} for event in triggers_raw}
        elif isinstance(triggers_raw, str):
            triggers = {triggers_raw: {}}
        elif isinstance(triggers_raw, dict):
            triggers = triggers_raw
        else:
            raise AssertionError(
                f"Unexpected trigger configuration in {workflow_file.name}: {type(triggers_raw)!r}"
            )

        trigger_keys = set(triggers)

        unexpected = sorted(trigger_keys & disallowed_triggers)
        assert not unexpected, (
            f"{workflow_file.name} exposes disallowed triggers: {unexpected}. "
            "Self-tests should not run automatically on PRs or pushes."
        )

        unsupported = sorted(trigger_keys - allowed_triggers)
        assert not unsupported, (
            f"{workflow_file.name} declares unsupported triggers: {unsupported}. "
            "Only workflow_dispatch, schedule, or workflow_call are permitted."
        )

        assert required_manual_trigger in trigger_keys, (
            f"{workflow_file.name} must provide a {required_manual_trigger} entry "
            "so self-tests remain manually invokable."
        )


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
    assert (
        SELFTEST_PATH.exists()
    ), "reusable-99-selftest.yml is missing from .github/workflows/"

    data = yaml.safe_load(SELFTEST_PATH.read_text())
    jobs = data.get("jobs", {})

    scenario_job = jobs.get("scenario")
    assert (
        scenario_job is not None
    ), "Scenario job missing from reusable-99-selftest.yml"
    assert (
        scenario_job.get("uses") == "./.github/workflows/reusable-ci.yml"
    ), "Scenario job must invoke reusable-ci.yml via jobs.<id>.uses"

    matrix = scenario_job.get("strategy", {}).get("matrix", {}).get("include", [])
    scenario_names = [entry.get("name") for entry in matrix]
    expected_names = [
        "minimal",
        "metrics_only",
        "metrics_history",
        "classification_only",
        "coverage_delta",
        "full_soft_gate",
    ]
    assert (
        scenario_names == expected_names
    ), "Self-test scenario matrix drifted; update verification docs/tests if intentional."

    aggregate_job = jobs.get("aggregate")
    assert aggregate_job is not None, "Aggregate verification job missing"
    assert (
        aggregate_job.get("needs") == "scenario"
    ), "Aggregate job must depend on the scenario matrix"
    assert (
        aggregate_job.get("if") == "${{ always() }}"
    ), "Aggregate job should always run to summarise results"

    permissions = aggregate_job.get("permissions", {})
    assert (
        permissions.get("actions") == "read"
    ), "Aggregate job must read workflow artifacts"
    assert (
        permissions.get("contents") == "read"
    ), "Aggregate job must read repository contents"

    env = aggregate_job.get("env", {})
    aggregate_list = env.get("SCENARIO_LIST", "")
    env_names = [name.strip() for name in aggregate_list.split(",") if name.strip()]
    assert (
        env_names == expected_names
    ), "Aggregate SCENARIO_LIST must stay aligned with the scenario matrix to keep summaries accurate."

    outputs = aggregate_job.get("outputs", {})
    assert {"verification_table", "failures"}.issubset(
        outputs
    ), "Aggregate job should surface verification outputs for downstream consumers."

    steps = aggregate_job.get("steps", [])
    verify_step = next((step for step in steps if step.get("id") == "verify"), None)
    assert (
        verify_step is not None
    ), "Aggregate job must include the github-script verification step"
    verify_env = verify_step.get("env", {})
    assert (
        verify_env.get("PYTHON_VERSIONS") == "${{ inputs.python-versions }}"
    ), "Verification step should read python-versions input via PYTHON_VERSIONS env var for dynamic artifact expectations."
