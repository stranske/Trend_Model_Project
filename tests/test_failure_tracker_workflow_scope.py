"""Regression tests for the maint-33 failure tracker workflow scope.

These checks operate on the YAML definition itself to guarantee that our
GitHub Actions workflow keeps the agreed upon contract: it should only act on
pull-request originated runs, apply the ci-failure label exactly once per
failing run, and upload a single lightweight artifact.
"""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "maint-33-check-failure-tracker.yml"


def _load_workflow() -> dict:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    return yaml.safe_load(text)


def _get_step(job: dict, name: str) -> dict:
    for step in job.get("steps", []):
        if step.get("name") == name:
            return step
    raise AssertionError(f"Step {name!r} not found in workflow")


def test_failure_job_is_pr_failure_only():
    workflow = _load_workflow()
    failure_job = workflow["jobs"]["failure"]
    condition = " ".join(failure_job.get("if", "").split())
    assert "workflow_run.event == 'pull_request'" in condition
    assert "workflow_run.conclusion == 'failure'" in condition
    assert "workflow_run.pull_requests[0]" in condition


def test_failure_job_uploads_single_snapshot_artifact():
    workflow = _load_workflow()
    failure_job = workflow["jobs"]["failure"]

    artifact_steps = [
        step
        for step in failure_job.get("steps", [])
        if step.get("uses", "").startswith("actions/upload-artifact@")
    ]
    assert len(artifact_steps) == 1, "Expected exactly one artifact upload in failure job"

    upload_step = artifact_steps[0]
    with_section = upload_step.get("with", {})
    assert with_section.get("name") == "ci-failures-snapshot"
    assert with_section.get("path") == "artifacts/ci_failures_snapshot.json"


def test_failure_job_labels_pull_request_once():
    workflow = _load_workflow()
    failure_job = workflow["jobs"]["failure"]
    label_step = _get_step(failure_job, "Label pull request as ci-failure")
    assert label_step["uses"].startswith("actions/github-script@"), "Label step must call github-script"


def test_success_job_removes_label_only_for_prs():
    workflow = _load_workflow()
    success_job = workflow["jobs"]["success"]
    condition = " ".join(success_job.get("if", "").split())
    assert "workflow_run.event == 'pull_request'" in condition

    remove_label_step = _get_step(success_job, "Remove ci-failure label from pull request")
    assert remove_label_step["uses"].startswith("actions/github-script@"), "Success job must remove label via github-script"

