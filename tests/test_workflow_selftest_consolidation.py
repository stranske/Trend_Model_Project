from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW_DIR = Path(".github/workflows")
ARCHIVE_DIR = Path("Old/workflows")
SELFTEST_PATH = WORKFLOW_DIR / "selftest-81-reusable-ci.yml"
RUNNER_PATH = WORKFLOW_DIR / "selftest-runner.yml"


def _resolve_triggers(data: dict) -> dict:
    """Normalize workflow `on` definitions to a mapping."""

    triggers_raw = data.get("on")
    if triggers_raw is None and True in data:
        triggers_raw = data[True]
    if triggers_raw is None:
        return {}

    if isinstance(triggers_raw, list):
        return {str(event): {} for event in triggers_raw}
    if isinstance(triggers_raw, str):
        return {triggers_raw: {}}
    if isinstance(triggers_raw, dict):
        return triggers_raw

    raise AssertionError(f"Unexpected trigger configuration: {type(triggers_raw)!r}")


def test_selftest_workflow_inventory() -> None:
    """Self-test workflows should match the expected manual roster."""

    selftest_workflows = sorted(
        path.name for path in WORKFLOW_DIR.glob("*selftest*.yml")
    )
    expected = [
        "selftest-81-reusable-ci.yml",
        "selftest-runner.yml",
    ]
    assert (
        selftest_workflows == expected
    ), f"Active self-test inventory drifted; expected {expected} but saw {selftest_workflows}."


def test_selftest_runner_inputs_cover_variants() -> None:
    """The consolidated runner should expose the requested input variants."""

    data = yaml.safe_load(RUNNER_PATH.read_text()) or {}
    triggers = _resolve_triggers(data)

    assert triggers, "selftest-runner.yml is missing trigger definitions."
    assert set(triggers) == {
        "workflow_dispatch"
    }, "Runner must remain a manual workflow_dispatch entry point."

    workflow_dispatch = triggers["workflow_dispatch"] or {}
    inputs = workflow_dispatch.get("inputs", {})

    def _assert_choice(
        field_name: str, expected_options: list[str], *, default: str | None
    ) -> None:
        field = inputs.get(field_name)
        assert (
            field is not None
        ), f"Missing `{field_name}` input on selftest-runner.yml."
        assert (
            field.get("type", "choice") == "choice"
        ), f"`{field_name}` should remain a choice input."
        options_raw = field.get("options", [])
        options_normalized = [str(option).lower() for option in options_raw]
        expected_normalized = [option.lower() for option in expected_options]
        assert (
            options_normalized == expected_normalized
        ), f"Unexpected option set for `{field_name}`: {options_raw!r}."
        if default is not None:
            actual_default = field.get("default")
            if isinstance(actual_default, bool):
                actual_default_normalized = str(actual_default).lower()
            else:
                actual_default_normalized = str(actual_default)
            assert (
                actual_default_normalized == default
            ), f"`{field_name}` default drifted from {default!r}."

    _assert_choice("mode", ["summary", "comment", "dual-runtime"], default="summary")
    _assert_choice("post_to", ["pr-number", "none"], default="none")
    _assert_choice("enable_history", ["true", "false"], default="false")

    pr_number = inputs.get("pull_request_number", {})
    required_raw = pr_number.get("required")
    assert required_raw in (
        None,
        False,
        "false",
    ), "pull_request_number must remain optional to reuse comment mode outside PRs."

    jobs = data.get("jobs", {})
    run_matrix = jobs.get("run-matrix") or {}
    assert (
        run_matrix.get("uses") == "./.github/workflows/selftest-81-reusable-ci.yml"
    ), "Runner should delegate execution to selftest-81-reusable-ci.yml."


def test_selftest_runner_publish_job_contract() -> None:
    """Publish-results job must enforce verification guardrails consistently."""

    data = yaml.safe_load(RUNNER_PATH.read_text()) or {}
    jobs = data.get("jobs", {})
    publish = jobs.get("publish-results") or {}

    assert publish, "selftest-runner.yml should retain the publish-results job."
    assert (
        publish.get("needs") == "run-matrix"
    ), "publish-results must depend on the reusable self-test matrix."
    assert (
        publish.get("if") == "${{ always() }}"
    ), "publish-results should always execute to surface matrix status."

    permissions = publish.get("permissions", {})
    assert permissions, "publish-results must declare minimal permissions."
    assert (
        permissions.get("contents") == "read"
    ), "publish-results should only require read access to contents."
    assert (
        permissions.get("actions") == "read"
    ), "publish-results should only require read access to actions metadata."
    assert (
        permissions.get("pull-requests") == "write"
    ), "publish-results needs pull request write access for comment mode."

    unexpected_permissions = sorted(
        key
        for key in permissions
        if key not in {"contents", "actions", "pull-requests"}
    )
    assert not unexpected_permissions, (
        "publish-results should not request extra permissions: "
        f"{unexpected_permissions}."
    )

    required_env = {
        "MODE",
        "POST_TO",
        "ENABLE_HISTORY",
        "PR_NUMBER",
        "SUMMARY_TITLE",
        "COMMENT_TITLE",
        "REASON",
        "WORKFLOW_RESULT",
        "VERIFICATION_TABLE",
        "FAILURE_COUNT",
        "RUN_ID",
        "REQUESTED_VERSIONS",
    }
    env = publish.get("env", {})
    missing_env = sorted(required_env - set(env))
    assert not missing_env, (
        "publish-results env block drifted; missing keys: " f"{missing_env}."
    )

    steps = publish.get("steps", [])

    def _find_step(name: str) -> dict:
        return next((step for step in steps if step.get("name") == name), {})

    download_step = _find_step("Download self-test report")
    assert download_step, "Download step missing from publish-results."
    assert (
        download_step.get("uses") == "actions/download-artifact@v4"
    ), "Download step should use actions/download-artifact@v4."
    assert (
        download_step.get("if")
        == "${{ env.ENABLE_HISTORY == 'true' && env.RUN_ID != '' }}"
    ), "Download step must guard on enable_history input and aggregate run id."
    download_with = download_step.get("with", {})
    assert (
        download_with.get("run-id") == "${{ env.RUN_ID }}"
    ), "Download step should forward the aggregate run id."
    assert (
        download_with.get("name") == "selftest-report"
    ), "Download step must keep artifact name stable for docs/tests."

    surface_failures = _find_step("Surface failures in logs")
    assert surface_failures, "Missing surface failures guard for summary mode."
    surface_script = surface_failures.get("run", "")
    for expected_snippet in (
        "Verification table output missing",
        "Failure count output missing",
        "Selftest runner reported",
        "Selftest matrix completed with status",
    ):
        assert (
            expected_snippet in surface_script
        ), f"Surface failures step should mention '{expected_snippet}'."

    comment_finalize = _find_step("Finalize status for comment mode")
    assert comment_finalize, "Comment mode finalizer missing."
    assert (
        comment_finalize.get("if") == "${{ env.MODE == 'comment' }}"
    ), "Comment finalizer should only run during comment mode."
    comment_script = comment_finalize.get("run", "")
    for snippet in (
        "Verification table output missing",
        "Failure count output missing",
        "Selftest runner reported",
        "Selftest matrix completed with status",
    ):
        assert (
            snippet in comment_script
        ), f"Comment finalizer guard should mention '{snippet}'."


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
        verify_env.get("PYTHON_VERSIONS") == "${{ inputs['python-versions'] }}"
    ), "Verification step should read python-versions input via PYTHON_VERSIONS env var for dynamic artifact expectations."


def test_selftest_reusable_exposes_workflow_call() -> None:
    data = yaml.safe_load(SELFTEST_PATH.read_text())
    triggers = data.get("on") or data.get(True) or {}
    assert (
        "workflow_call" in triggers
    ), "selftest-81-reusable-ci.yml must expose workflow_call for downstream wrappers."
