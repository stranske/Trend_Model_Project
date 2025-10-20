"""Structural tests covering the Codex belt workflow pipeline.

These checks validate that the dispatcher, worker, and conveyor workflows
retain the critical wiring described in issue #2853. They guard the
automation pipeline against regressions by ensuring the YAML definitions keep
the PAT guards, dispatch wiring, and re-dispatch behaviour that Acceptance
Criteria rely upon.
"""

from __future__ import annotations

import pathlib
from typing import Any

import yaml


WORKFLOW_ROOT = pathlib.Path(".github/workflows")


def _normalise_keys(node: Any) -> Any:
    if isinstance(node, dict):
        normalised: dict[str, Any] = {}
        for key, value in node.items():
            match key:
                case bool() as boolean:
                    key_str = "on" if boolean else str(boolean).lower()
                case str() as text:
                    key_str = text
                case other:
                    key_str = str(other)
            normalised[key_str] = _normalise_keys(value)
        return normalised
    if isinstance(node, list):
        return [_normalise_keys(item) for item in node]
    return node


def _load_workflow(slug: str) -> dict[str, Any]:
    path = WORKFLOW_ROOT / slug
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    assert isinstance(
        data, dict
    ), f"Workflow {slug} should load into a mapping structure"
    return _normalise_keys(data)


def _step_runs_command(step: dict[str, Any], needle: str) -> bool:
    script = step.get("run") or ((step.get("with") or {}).get("script"))
    if not isinstance(script, str):
        return False
    return needle in script


def test_dispatcher_wires_repo_dispatch_event():
    workflow = _load_workflow("agents-71-codex-belt-dispatcher.yml")
    jobs = workflow.get("jobs") or {}
    dispatch_job = jobs.get("dispatch") or {}
    steps = dispatch_job.get("steps") or []
    assert steps, "Dispatcher job must define steps"

    guard = steps[0]
    assert guard.get("name") == "Ensure ACTIONS_BOT_PAT is configured"
    assert _step_runs_command(
        guard, "ACTIONS_BOT_PAT secret is required for dispatcher writes."
    ), "Dispatcher must fail early when ACTIONS_BOT_PAT is missing"

    dispatch_steps = [step for step in steps if step.get("name") == "Dispatch worker"]
    assert dispatch_steps, "Dispatcher must hand off work to the worker"
    worker_step = dispatch_steps[0]
    script = (worker_step.get("with") or {}).get("script", "")
    assert "createDispatchEvent" in script
    assert "codex-belt.work" in script


def test_worker_keeps_concurrency_and_pat_guard():
    workflow = _load_workflow("agents-72-codex-belt-worker.yml")

    concurrency = workflow.get("concurrency") or {}
    assert concurrency.get("group") == "codex-belt"
    assert concurrency.get("cancel-in-progress") is False

    events = workflow.get("on") or {}
    repo_dispatch = events.get("repository_dispatch") or {}
    types = repo_dispatch.get("types") or []
    assert "codex-belt.work" in types

    jobs = workflow.get("jobs") or {}
    bootstrap = jobs.get("bootstrap") or {}
    steps = bootstrap.get("steps") or []
    assert steps, "Worker bootstrap job must define steps"
    guard = steps[0]
    assert guard.get("name") == "Ensure ACTIONS_BOT_PAT is configured"
    assert _step_runs_command(
        guard, "ACTIONS_BOT_PAT secret is required for worker actions."
    )


def test_conveyor_requires_gate_success_and_retriggers_dispatcher():
    workflow = _load_workflow("agents-73-codex-belt-conveyor.yml")
    jobs = workflow.get("jobs") or {}
    promote = jobs.get("promote") or {}
    condition = promote.get("if") or ""
    assert "workflow_run.conclusion == 'success'" in condition
    assert "workflow_run.event == 'pull_request'" in condition

    steps = promote.get("steps") or []
    assert steps, "Conveyor promote job must define steps"

    guard = steps[0]
    assert guard.get("name") == "Ensure ACTIONS_BOT_PAT is configured"
    assert _step_runs_command(
        guard, "ACTIONS_BOT_PAT secret is required for conveyor actions."
    )

    redispatch_steps = [
        step for step in steps if step.get("name") == "Re-dispatch dispatcher"
    ]
    assert redispatch_steps, "Conveyor must re-trigger the dispatcher"
    redispatch = redispatch_steps[0]
    script = (redispatch.get("with") or {}).get("script", "")
    assert "createWorkflowDispatch" in script
    assert "agents-71-codex-belt-dispatcher.yml" in script
