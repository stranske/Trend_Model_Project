from __future__ import annotations

import pathlib
from typing import Any, Dict, Iterable

import yaml

WORKFLOWS = pathlib.Path(".github/workflows")
WORKFLOW_FILE = "maint-46-post-ci.yml"


def _load_workflow(name: str) -> Dict[str, Any]:
    with (WORKFLOWS / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _step_by_name(steps: Iterable[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for step in steps:
        if step.get("name") == name:
            return step
    raise AssertionError(f"Expected workflow step '{name}' not found")


def _job_steps(data: Dict[str, Any], job: str) -> Iterable[Dict[str, Any]]:
    job_def = data["jobs"][job]
    steps = job_def.get("steps")
    if steps is None:
        raise KeyError("steps")
    return steps


def test_gate_rerun_job_requires_autofix_changes() -> None:
    data = _load_workflow(WORKFLOW_FILE)
    job_def = data["jobs"]["gate-rerun"]
    condition = job_def.get("if", "")
    assert "needs.small-fixes-meta.outputs.changed == 'true'" in condition
    assert "needs.context.outputs.same_repo == 'true'" in condition


def test_autofix_comment_job_requires_same_repo_and_changes() -> None:
    data = _load_workflow(WORKFLOW_FILE)
    job_def = data["jobs"]["autofix-comment"]
    condition = job_def.get("if", "")
    assert "needs.context.outputs.same_repo == 'true'" in condition
    assert "needs.small-fixes-meta.outputs.changed == 'true'" in condition


def test_consolidated_comment_includes_patch_instructions() -> None:
    data = _load_workflow(WORKFLOW_FILE)
    steps = _job_steps(data, "post-comment")
    prepare_step = _step_by_name(steps, "Prepare consolidated comment")
    script = prepare_step.get("run", "")
    assert "git am < autofix.patch" in script
    assert "Patch artifact:" in script


def test_autofix_opt_in_label_normalized_to_clean() -> None:
    data = _load_workflow(WORKFLOW_FILE)
    root_env = data.get("env", {})
    assert root_env.get("AUTOFIX_LABEL") == "autofix:clean"

    context_env = data["jobs"]["context"]["env"]
    assert (
        context_env["AUTOFIX_LABEL"] == "${{ env.AUTOFIX_LABEL }}"
    ), "Context job must source AUTOFIX_LABEL from the workflow environment"

    small_with = data["jobs"]["small-fixes"]["with"]
    assert (
        small_with["opt_in_label"] == "${{ env.AUTOFIX_LABEL }}"
    ), "Small fixes job must forward AUTOFIX_LABEL as the opt-in label"
    assert (
        small_with["clean_label"] == "${{ env.AUTOFIX_LABEL }}"
    ), "Clean label should mirror the opt-in label"
    assert (
        small_with["applied_label"] == "${{ env.AUTOFIX_APPLIED_LABEL }}"
    ), "Applied label should reference AUTOFIX_APPLIED_LABEL"
    assert (
        small_with["patch_label"] == "${{ env.AUTOFIX_PATCH_LABEL }}"
    ), "Patch label should reference AUTOFIX_PATCH_LABEL"
    assert (
        small_with["dry_run"] == "${{ needs.context.outputs.same_repo != 'true' }}"
    ), "Small fixes job must forward an explicit dry_run toggle for fork safety"
