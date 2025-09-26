from __future__ import annotations

import pathlib
from typing import Any, Dict, Iterable

import pytest

import yaml

WORKFLOWS = pathlib.Path(".github/workflows")


def _load_workflow(name: str) -> Dict[str, Any]:
    with (WORKFLOWS / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _step_by_name(steps: Iterable[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for step in steps:
        if step.get("name") == name:
            return step
    raise AssertionError(f"Expected workflow step '{name}' not found")


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
    data = _load_workflow("autofix.yml")
    steps = data["jobs"][job]["steps"]

    comment_step = _step_by_name(steps, "Comment with patch instructions (fork)")
    condition = comment_step.get("if", "")
    assert "same_repo != 'true'" in condition
    assert f"steps.{changed_step}.outputs.changed == 'true'" in condition
    assert comment_step.get("uses", "").startswith(
        "actions/github-script@"
    ), "Fork comment step should use github-script to leave instructions"

    summary_step = _step_by_name(steps, "Summary")
    run_script = summary_step.get("run", "")
    expected_line = "Patch artifact: autofix-patch-pr-${{ needs.context.outputs.pr }}"
    assert expected_line in run_script
    assert "${{ needs.context.outputs.same_repo }}" in run_script
    assert "${{ steps." in run_script
