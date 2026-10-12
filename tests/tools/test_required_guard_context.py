"""Pin the repo-local declaration to the unchanged shared guard producer."""

from pathlib import Path

import yaml
import pytest

from tools.enforce_gate_branch_protection import DEFAULT_CONTEXTS, load_required_contexts, main

ROOT = Path(__file__).resolve().parents[2]


def test_cli_context_help_matches_required_guard_names(capsys) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "'Gate / gate', 'guard', and 'Health 45 Agents Guard / guard'" in help_text
    assert "Enforce agents workflow protections" not in help_text


def test_required_guard_context_matches_emitted_job_name() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/agents-guard.yml").read_text())
    job = workflow["jobs"]["guard"]
    emitted_name = job.get("name", "guard")
    declared = load_required_contexts(ROOT / ".github/config/required-contexts.json")
    assert emitted_name == "guard"
    assert declared == ["Gate / gate", emitted_name, "Health 45 Agents Guard / guard"]
    assert list(DEFAULT_CONTEXTS) == declared


def test_guard_failure_still_fails_the_producer() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/agents-guard.yml").read_text())
    job = workflow["jobs"]["guard"]
    assert not job.get("continue-on-error", False)
    failure_step = next(
        step for step in job["steps"] if step["name"] == "Fail when guard blocks the pull request"
    )
    assert "steps.evaluate.outputs.blocked == 'true'" in failure_step["if"]
    assert "exit 1" in failure_step["run"]
    assert not failure_step.get("continue-on-error", False)


def test_label_change_guard_status_is_bound_to_pr_head() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/agents-guard.yml").read_text())
    report = next(
        step
        for step in workflow["jobs"]["guard"]["steps"]
        if step["name"] == "Report agents guard commit status"
    )
    script = report["with"]["script"]
    assert "pullRequest?.head?.sha || context.sha" in script
    assert "context: 'Health 45 Agents Guard / guard'" in script
    assert "blocked ? 'failure' : 'success'" in script
    assert "Health 45 Agents Guard / guard" in load_required_contexts(
        ROOT / ".github/config/required-contexts.json"
    )
