from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW_DIR = Path(".github/workflows")
ARCHIVE_DIR = Path("Old/workflows")
SELFTEST_PATH = WORKFLOW_DIR / "selftest-81-reusable-ci.yml"


def test_selftest_workflow_inventory() -> None:
    """Self-test workflows should match the expected manual roster."""

    selftest_workflows = sorted(
        path.name for path in WORKFLOW_DIR.glob("*selftest*.yml")
    )
    expected = [
        "selftest-80-pr-comment.yml",
        "selftest-81-reusable-ci.yml",
        "selftest-82-pr-comment.yml",
        "selftest-83-pr-comment.yml",
        "selftest-84-reusable-ci.yml",
        "selftest-88-reusable-ci.yml",
    ]
    assert (
        selftest_workflows == expected
    ), f"Active self-test inventory drifted; expected {expected} but saw {selftest_workflows}."


def test_selftest_triggers_are_manual_only() -> None:
    """Self-test workflows must expose manual triggers (with optional
    workflow_call reuse)."""

    selftest_files = sorted(WORKFLOW_DIR.glob("*selftest*.yml"))
    assert selftest_files, "Expected at least one self-test workflow definition."

    disallowed_triggers = {
        "pull_request",
        "pull_request_target",
        "push",
        "schedule",
    }
    required_manual_trigger = "workflow_dispatch"
    allowed_triggers = {required_manual_trigger, "workflow_call"}

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
            "Self-tests should not run automatically on PRs, pushes, or schedules."
        )

        unsupported = sorted(trigger_keys - allowed_triggers)
        assert not unsupported, (
            f"{workflow_file.name} declares unsupported triggers: {unsupported}. "
            "Only workflow_dispatch (and workflow_call for reusable entry points) are permitted."
        )

        assert required_manual_trigger in trigger_keys, (
            f"{workflow_file.name} must provide a {required_manual_trigger} "
            "trigger so self-tests remain manually invoked."
        )

        if workflow_file.name != "selftest-81-reusable-ci.yml":
            assert trigger_keys == {required_manual_trigger}, (
                f"{workflow_file.name} should only expose {required_manual_trigger}. "
                "Workflow-call support is reserved for selftest-81-reusable-ci.yml."
            )


def test_selftest_dispatch_reason_input() -> None:
    """Self-test dispatch should keep the reason field optional but present."""

    raw_data = yaml.safe_load(SELFTEST_PATH.read_text()) or {}

    triggers_raw = raw_data.get("on")
    if triggers_raw is None and True in raw_data:
        triggers_raw = raw_data[True]
    if triggers_raw is None:
        triggers_raw = {}

    if isinstance(triggers_raw, dict):
        workflow_dispatch = triggers_raw.get("workflow_dispatch", {})
    else:
        raise AssertionError(
            "Selftest workflow must declare workflow_dispatch as a mapping"
        )

    inputs = workflow_dispatch.get("inputs", {})
    assert "reason" in inputs, "Selftest workflow dispatch is missing a reason input."

    reason_input = inputs["reason"] or {}
    # Normalize the "required" field to a boolean for consistency.
    required_raw = reason_input.get("required")
    required_bool = False if required_raw in (False, "false", None) else True
    assert (
        not required_bool
    ), "Selftest workflow dispatch reason should be optional for manual launches."

    description = reason_input.get("description", "").strip()
    assert description, "Reason input should document why it is collected."

    default_value = reason_input.get("default")
    assert (
        default_value == "manual test"
    ), "Reason input default should remain 'manual test' so callers understand the context."


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


def test_archived_selftests_retain_manual_triggers() -> None:
    """Archived self-test wrappers should stay manual-first to avoid
    regressions."""

    archived_files = sorted(ARCHIVE_DIR.glob("*selftest*.yml"))
    assert (
        archived_files
    ), "Expected archived self-test workflows to remain in Old/workflows/."

    disallowed_triggers = {
        "pull_request",
        "pull_request_target",
        "push",
        "schedule",
    }
    required_manual_trigger = "workflow_dispatch"
    optional_triggers = {"workflow_call"}
    allowed_triggers = {required_manual_trigger} | optional_triggers

    for workflow_file in archived_files:
        data = yaml.safe_load(workflow_file.read_text()) or {}

        triggers_raw = data.get("on")
        if triggers_raw is None and True in data:
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
            "Archived self-tests should remain manual-only entry points (no PR, push, or scheduled automation)."
        )

        unsupported = sorted(trigger_keys - allowed_triggers)
        assert not unsupported, (
            f"{workflow_file.name} declares unsupported triggers: {unsupported}. "
            "Only workflow_dispatch or workflow_call are permitted."
        )

        assert required_manual_trigger in trigger_keys, (
            f"{workflow_file.name} must retain a {required_manual_trigger} entry "
            "so the wrapper can be invoked manually if restored."
        )


def test_selftest_matrix_and_aggregate_contract() -> None:
    assert (
        SELFTEST_PATH.exists()
    ), "selftest-81-reusable-ci.yml is missing from .github/workflows/"

    data = yaml.safe_load(SELFTEST_PATH.read_text())
    jobs = data.get("jobs", {})

    scenario_job = jobs.get("scenario")
    assert (
        scenario_job is not None
    ), "Scenario job missing from selftest-81-reusable-ci.yml"
    assert (
        scenario_job.get("uses") == "./.github/workflows/reusable-10-ci-python.yml"
    ), "Scenario job must invoke reusable-10-ci-python.yml via jobs.<id>.uses"

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
    assert {"verification_table", "failures", "run_id"}.issubset(
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


def test_selftest_reusable_exposes_workflow_call() -> None:
    data = yaml.safe_load(SELFTEST_PATH.read_text())
    triggers = data.get("on") or data.get(True) or {}
    assert (
        "workflow_call" in triggers
    ), "selftest-81-reusable-ci.yml must expose workflow_call for downstream wrappers."
