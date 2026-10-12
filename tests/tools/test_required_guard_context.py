"""Pin the repo-local declaration to the unchanged shared guard producer."""

from pathlib import Path

import yaml

from tools.enforce_gate_branch_protection import DEFAULT_CONTEXTS, load_required_contexts

ROOT = Path(__file__).resolve().parents[2]


def test_required_guard_context_matches_emitted_job_name() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/agents-guard.yml").read_text())
    job = workflow["jobs"]["guard"]
    emitted_name = job.get("name", "guard")
    declared = load_required_contexts(ROOT / ".github/config/required-contexts.json")
    assert emitted_name == "guard"
    assert declared == ["Gate / gate", emitted_name]
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
