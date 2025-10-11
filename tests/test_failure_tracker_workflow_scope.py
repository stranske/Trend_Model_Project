"""Regression tests for the failure tracker delegation pipeline."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKER_PATH = (
    REPO_ROOT / ".github" / "workflows" / "maint-33-check-failure-tracker.yml"
)
POST_CI_PATH = REPO_ROOT / ".github" / "workflows" / "maint-post-ci.yml"


def _load_workflow(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text)


def _get_step(job: dict, name: str) -> dict:
    for step in job.get("steps", []):
        if step.get("name") == name:
            return step
    raise AssertionError(f"Step {name!r} not found in workflow")


def test_tracker_workflow_is_now_thin_shell() -> None:
    workflow = _load_workflow(TRACKER_PATH)
    assert set(workflow["jobs"].keys()) == {"redirect"}
    redirect_job = workflow["jobs"]["redirect"]
    condition = " ".join(redirect_job.get("if", "").split())
    assert "workflow_run.event == 'pull_request'" in condition
    summary_step = _get_step(redirect_job, "Emit delegation summary")
    summary_body = summary_step.get("run", "")
    assert "maint-post-ci.yml" in summary_body


def test_post_ci_failure_tracker_handles_failure_path() -> None:
    workflow = _load_workflow(POST_CI_PATH)
    job = workflow["jobs"]["failure-tracker"]
    condition = " ".join(job.get("if", "").split())
    assert "needs.context.outputs.found == 'true'" in condition
    assert "workflow_run.event == 'pull_request'" in condition

    tracker_step = _get_step(job, "Derive failure signature & update tracking issue")
    assert tracker_step["if"].strip() == "github.event.workflow_run.conclusion == 'failure'"

    label_step = _get_step(job, "Label pull request as ci-failure")
    assert label_step["uses"].startswith("actions/github-script@")

    artifact_steps = [
        step
        for step in job.get("steps", [])
        if step.get("uses", "").startswith("actions/upload-artifact@")
    ]
    assert len(artifact_steps) == 2, "Failure and success paths should each upload once"
    for step in artifact_steps:
        with_section = step.get("with", {})
        assert with_section.get("name") == "ci-failures-snapshot"
        assert with_section.get("path") == "artifacts/ci_failures_snapshot.json"


def test_post_ci_failure_tracker_handles_success_path() -> None:
    workflow = _load_workflow(POST_CI_PATH)
    job = workflow["jobs"]["failure-tracker"]

    heal_step = _get_step(job, "Auto-heal stale failure issues & note success")
    assert heal_step["if"].strip() == "github.event.workflow_run.conclusion == 'success'"

    remove_label_step = _get_step(job, "Remove ci-failure label from pull request")
    assert remove_label_step["uses"].startswith("actions/github-script@")
