from __future__ import annotations

import pathlib
from typing import Any, Dict, Iterable

import pytest

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


@pytest.mark.parametrize(
    ("job", "changed_step"),
    (
        ("small-fixes", "apply"),
        ("fix-failing-checks", "autofix"),
    ),
)
def test_autofix_remote_repo_path_posts_patch_instructions(
    job: str, changed_step: str
) -> None:
    data = _load_workflow(WORKFLOW_FILE)
    job_def = data["jobs"][job]

    steps = job_def.get("steps")

    if steps is not None:
        if job == "small-fixes":
            label_step = _step_by_name(steps, "Label PR (autofix patch available)")
            condition = label_step.get("if", "")
            assert "same_repo != 'true'" in condition
            assert f"steps.{changed_step}.outputs.changed == 'true'" in condition
            assert label_step.get("uses", "").startswith(
                "actions/github-script@"
            ), "Fork patch label step should use github-script to interact with the PR"
        else:
            assert not any(
                step.get("name") == "Label PR (autofix patch available)" for step in steps
            ), "Patch label step should not exist in fix-failing-checks job"
    else:
        assert (
            job_def.get("uses") == "./.github/workflows/reusable-18-autofix.yml"
        ), "Small fixes job should invoke the reusable autofix workflow"
        return

    summary_step = _step_by_name(steps, "Summary")
    run_script = summary_step.get("run", "")
    if job == "small-fixes":
        assert "Patch artifact:" in run_script
        assert "${GITHUB_RUN_ID}" in run_script
        assert "#artifacts" in run_script
        assert "${{ needs.context.outputs.same_repo }}" in run_script
        assert "${{ steps." in run_script
    else:
        assert "Fix failing checks" in run_script
        assert "${{ steps." in run_script


def test_consolidated_comment_includes_patch_instructions() -> None:
    data = _load_workflow(WORKFLOW_FILE)
    steps = _job_steps(data, "post-comment")
    prepare_step = _step_by_name(steps, "Prepare consolidated comment")
    script = prepare_step.get("run", "")
    assert "git am < autofix.patch" in script
    assert "Patch artifact:" in script


def test_autofix_opt_in_label_normalized_to_clean() -> None:
    data = _load_workflow(WORKFLOW_FILE)
    context_env = data["jobs"]["context"]["env"]
    assert "autofix:clean" in context_env["AUTOFIX_OPT_IN_LABEL"], (
        "Context job must default AUTOFIX_OPT_IN_LABEL to autofix:clean"
    )

    small_with = data["jobs"]["small-fixes"]["with"]
    assert "autofix:clean" in small_with["opt_in_label"], (
        "Small fixes job must forward autofix:clean as the opt-in label"
    )
    assert (
        small_with["clean_label"].count("autofix:clean") == 1
        and "autofix:clean" in small_with["clean_label"]
    ), "Clean label should mirror the opt-in label"
