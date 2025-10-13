"""Regression tests for the failure tracker delegation pipeline."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKER_PATH = (
    REPO_ROOT / ".github" / "workflows" / "maint-47-check-failure-tracker.yml"
)
POST_CI_PATH = REPO_ROOT / ".github" / "workflows" / "maint-46-post-ci.yml"


def _load_workflow(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text)


def _get_step(job: dict, name: str) -> dict:
    for step in job.get("steps", []):
        if step.get("name") == name:
            return step
    raise AssertionError(f"Step {name!r} not found in workflow")


def _get_tracker_script() -> str:
    workflow = _load_workflow(POST_CI_PATH)
    job = workflow["jobs"]["failure-tracker"]
    tracker_step = _get_step(job, "Derive failure signature & update tracking issue")
    script = tracker_step.get("with", {}).get("script")
    assert isinstance(script, str), "Expected inline tracker script to be defined"
    return script


def test_tracker_workflow_is_now_thin_shell() -> None:
    workflow = _load_workflow(TRACKER_PATH)
    assert set(workflow["jobs"].keys()) == {"redirect"}
    redirect_job = workflow["jobs"]["redirect"]
    condition = " ".join(redirect_job.get("if", "").split())
    assert "workflow_run.event == 'pull_request'" in condition
    summary_step = _get_step(redirect_job, "Emit delegation summary")
    summary_body = summary_step.get("run", "")
    assert "maint-46-post-ci.yml" in summary_body


def test_tracker_shell_performs_no_issue_writes() -> None:
    workflow = _load_workflow(TRACKER_PATH)
    redirect_job = workflow["jobs"]["redirect"]

    for step in redirect_job.get("steps", []):
        assert (
            step.get("uses") is None
        ), "Delegation shell should not invoke external actions"
        script = step.get("run", "")
        assert "github.rest.issues" not in script


def test_tracker_workflow_triggers_from_gate_run() -> None:
    workflow = _load_workflow(TRACKER_PATH)

    trigger = workflow.get("on")
    assert trigger is not None, "Expected workflow_run trigger to be defined"

    workflow_run = trigger.get("workflow_run")
    assert (
        workflow_run is not None
    ), "Failure tracker should listen to workflow_run events"

    assert workflow_run.get("types") == ["completed"]
    assert workflow_run.get("workflows") == ["Gate"]


def test_maint_post_ci_listens_to_gate_run() -> None:
    workflow = _load_workflow(POST_CI_PATH)

    trigger = workflow.get("on")
    assert trigger is not None, "Maint 46 Post CI should define workflow_run trigger"

    workflow_run = trigger.get("workflow_run")
    assert (
        workflow_run is not None
    ), "Maint 46 Post CI should listen to workflow_run events"

    assert workflow_run.get("types") == ["completed"]
    assert workflow_run.get("workflows") == ["Gate"]


def test_context_exposes_failure_tracker_skip_for_legacy_prs() -> None:
    workflow = _load_workflow(POST_CI_PATH)
    context_job = workflow["jobs"]["context"]

    outputs = context_job.get("outputs", {})
    assert (
        outputs.get("failure_tracker_skip")
        == "${{ steps.info.outputs.failure_tracker_skip }}"
    )

    info_step = _get_step(context_job, "Resolve workflow context")
    script = info_step.get("with", {}).get("script", "")
    assert "new Set([10, 12])" in script


def test_post_ci_failure_tracker_handles_failure_path() -> None:
    workflow = _load_workflow(POST_CI_PATH)
    job = workflow["jobs"]["failure-tracker"]
    condition = " ".join(job.get("if", "").split())
    assert "needs.context.outputs.found == 'true'" in condition
    assert "needs.context.outputs.failure_incomplete != 'true'" in condition
    assert "needs.context.outputs.failure_tracker_skip != 'true'" in condition
    assert "workflow_run.event == 'pull_request'" in condition

    tracker_step = _get_step(job, "Derive failure signature & update tracking issue")
    assert (
        tracker_step["if"].strip()
        == "github.event.workflow_run.conclusion == 'failure'"
    )

    summary_step = _get_step(job, "Emit failure summary")
    assert (
        summary_step["if"].strip()
        == "github.event.workflow_run.conclusion == 'failure'"
    ), "Failure summary should only emit on failing runs"

    label_step = _get_step(job, "Label pull request as ci-failure")
    assert label_step["uses"].startswith("actions/github-script@")
    label_condition = label_step.get("if", "").strip()
    assert "github.event.workflow_run.conclusion == 'failure'" in label_condition
    label_script = label_step.get("with", {}).get("script", "")
    assert "'ci-failure'" in label_script
    assert "github.rest.issues.addLabels" in label_script

    tracker_script = tracker_step.get("with", {}).get("script", "")
    assert "github.rest.issues.update({" in tracker_script
    assert "github.rest.issues.createComment" in tracker_script

    artifact_steps = [
        step
        for step in job.get("steps", [])
        if step.get("uses", "").startswith("actions/upload-artifact@")
    ]
    assert len(artifact_steps) == 2, "Failure and success paths should each upload once"

    seen_conditions: set[str] = set()
    for step in artifact_steps:
        with_section = step.get("with", {})
        assert with_section.get("name") == "ci-failures-snapshot"
        assert with_section.get("path") == "artifacts/ci_failures_snapshot.json"

        condition = " ".join(step.get("if", "").split())
        if condition:
            seen_conditions.add(condition)

    assert (
        "github.event.workflow_run.conclusion == 'failure'" in seen_conditions
    ), "Failure artifact upload should be gated on failing runs"
    assert (
        "github.event.workflow_run.conclusion == 'success'" in seen_conditions
    ), "Success artifact upload should mirror failure artifact payload"


def test_post_ci_failure_tracker_handles_success_path() -> None:
    workflow = _load_workflow(POST_CI_PATH)
    job = workflow["jobs"]["failure-tracker"]

    heal_step = _get_step(job, "Auto-heal stale failure issues & note success")
    assert (
        heal_step["if"].strip() == "github.event.workflow_run.conclusion == 'success'"
    )

    remove_label_step = _get_step(job, "Remove ci-failure label from pull request")
    assert remove_label_step["uses"].startswith("actions/github-script@")
    remove_condition = remove_label_step.get("if", "").strip()
    assert "github.event.workflow_run.conclusion == 'success'" in remove_condition
    remove_script = remove_label_step.get("with", {}).get("script", "")
    assert "ci-failure" in remove_script
    assert "github.rest.issues.removeLabel" in remove_script


def test_post_ci_requires_issue_permissions() -> None:
    workflow = _load_workflow(POST_CI_PATH)
    permissions = workflow.get("permissions", {})

    assert permissions.get("issues") == "write"
    assert permissions.get("pull-requests") == "write"


def test_post_comment_job_upserts_single_pr_comment() -> None:
    workflow = _load_workflow(POST_CI_PATH)
    job = workflow["jobs"]["post-comment"]

    condition = " ".join(job.get("if", "").split())
    assert "needs.context.result == 'success'" in condition
    assert "needs.context.outputs.found == 'true'" in condition
    assert "needs.context.outputs.failure_tracker_skip != 'true'" in condition
    assert "needs.context.outputs.failure_incomplete != 'true'" in condition

    comment_step = _get_step(job, "Upsert consolidated PR comment")
    script = comment_step.get("with", {}).get("script", "")

    assert "<!-- maint-46-post-ci: DO NOT EDIT -->" in script
    assert "github.rest.issues.updateComment" in script
    assert "github.rest.issues.createComment" in script


def test_failure_tracker_signature_uses_slugified_workflow_path() -> None:
    script = _get_tracker_script()

    assert "const slugify = (value) =>" in script
    assert "const workflowPath = (run.path || '').trim();" in script
    assert (
        "const workflowStem = workflowFile.replace(/\\.ya?ml$/i, '') || workflowFile;"
        in script
    )
    assert (
        "const workflowIdRaw = workflowStem || workflowPath || rawWorkflowName || '';"
        in script
    )
    assert "const workflowId = slugify(workflowIdRaw);" in script
    assert (
        "const fallbackId = run.workflow_id ? `workflow-${run.workflow_id}` : (run.id ? `run-${run.id}` : 'workflow');"
        in script
    )
    assert "const workflowKey = workflowId || fallbackId;" in script
    assert "const signature = `${workflowKey}|${sigHash}`;" in script
    assert "Workflow slug: \\`${workflowId}\\`" in script
