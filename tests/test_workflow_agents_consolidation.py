from pathlib import Path

import yaml

WORKFLOWS_DIR = Path(".github/workflows")


def _load_workflow_yaml(name: str) -> dict:
    path = WORKFLOWS_DIR / name
    assert path.exists(), f"Workflow {name} must exist"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _workflow_on_section(data: dict) -> dict:
    return data.get("on") or data.get(True) or {}


def test_agents_orchestrator_inputs_and_uses():
    wf = WORKFLOWS_DIR / "agents-70-orchestrator.yml"
    assert wf.exists(), "agents-70-orchestrator.yml must exist"
    text = wf.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text, "Orchestrator must allow manual dispatch"
    expected_inputs = {
        "enable_readiness",
        "readiness_agents",
        "require_all",
        "enable_preflight",
        "codex_user",
        "enable_verify_issue",
        "verify_issue_number",
        "enable_watchdog",
        "draft_pr",
        "options_json",
    }
    for key in expected_inputs:
        assert f"{key}:" in text, f"Missing workflow_dispatch input: {key}"
    assert (
        "fromJson(inputs.options_json || '{}')" in text
    ), "options_json must be parsed via fromJson()"
    assert (
        "enable_bootstrap:" in text
    ), "Orchestrator must forward enable_bootstrap flag"
    assert (
        "bootstrap_issues_label:" in text
    ), "Orchestrator must forward bootstrap label configuration"
    assert (
        "./.github/workflows/reusable-70-agents.yml" in text
    ), "Orchestrator must call the reusable agents workflow"


def test_reusable_agents_workflow_structure():
    reusable = WORKFLOWS_DIR / "reusable-70-agents.yml"
    assert reusable.exists(), "reusable-70-agents.yml must exist"
    text = reusable.read_text(encoding="utf-8")
    assert "workflow_call:" in text, "Reusable agents workflow must be callable"
    for key in [
        "readiness_custom_logins",
        "require_all",
        "enable_preflight",
        "enable_verify_issue",
        "enable_watchdog",
        "enable_keepalive",
        "options_json",
    ]:
        assert f"{key}:" in text, f"Reusable agents workflow must expose input: {key}"


def test_legacy_agent_workflows_removed():
    present = {p.name for p in WORKFLOWS_DIR.glob("agents-*.yml")}
    forbidden = {
        "agents-40-consumer.yml",
        "agents-41-assign-and-watch.yml",
        "agents-41-assign.yml",
        "agents-42-watchdog.yml",
        "agents-44-copilot-readiness.yml",
        "agents-45-verify-codex-bootstrap-matrix.yml",
    }
    assert not (
        present & forbidden
    ), f"Legacy agent workflows still present: {present & forbidden}"


def test_codex_issue_bridge_present():
    bridge = WORKFLOWS_DIR / "agents-43-codex-issue-bridge.yml"
    assert (
        bridge.exists()
    ), "agents-43-codex-issue-bridge.yml must exist after Codex bridge restoration"


def test_keepalive_job_present():
    reusable = WORKFLOWS_DIR / "reusable-70-agents.yml"
    text = reusable.read_text(encoding="utf-8")
    assert (
        "Codex Keepalive Sweep" in text
    ), "Keepalive job must exist in reusable agents workflow"
    assert (
        "enable_keepalive" in text
    ), "Keepalive job must document enable_keepalive option"
    assert (
        "<!-- codex-keepalive -->" in text
    ), "Keepalive marker must be retained for duplicate suppression"
    assert (
        "issue_numbers_json" in text
    ), "Ready issues step must emit issue_numbers_json output"
    assert "first_issue" in text, "Ready issues step must emit first_issue output"


def test_agents_orchestrator_has_concurrency_defaults():
    data = _load_workflow_yaml("agents-70-orchestrator.yml")

    concurrency = data.get("concurrency") or {}
    assert (
        concurrency.get("group") == "agents-orchestrator-${{ github.ref }}"
    ), "Orchestrator must serialize runs per ref"
    assert (
        concurrency.get("cancel-in-progress") is True
    ), "Orchestrator concurrency must cancel in-progress runs"

    jobs = data.get("jobs", {})
    orchestrate = jobs.get("orchestrate", {})
    assert orchestrate.get("uses"), "Orchestrator job should call the reusable workflow"
    assert (
        "timeout-minutes" not in orchestrate
    ), "Timeout must live in reusable workflow because workflow-call jobs reject timeout-minutes"

    text = (WORKFLOWS_DIR / "agents-70-orchestrator.yml").read_text(encoding="utf-8")
    assert (
        "Job timeouts live inside reusable-70-agents.yml" in text
    ), "Orchestrator workflow should document where the timeout is enforced"


def test_agents_consumer_manual_only_and_concurrency():
    data = _load_workflow_yaml("agents-62-consumer.yml")

    on_section = _workflow_on_section(data)
    assert set(on_section.keys()) == {
        "workflow_dispatch"
    }, "Agents Consumer must only allow manual workflow_dispatch runs"

    concurrency = data.get("concurrency") or {}
    assert (
        concurrency.get("group") == "agents-62-consumer-${{ github.ref }}"
    ), "Agents Consumer concurrency group must scope runs by ref"
    assert (
        concurrency.get("cancel-in-progress") is True
    ), "Agents Consumer concurrency must cancel in-progress runs"

    jobs = data.get("jobs", {})
    dispatch_job = jobs.get("dispatch", {})
    assert dispatch_job.get("uses"), "Agents Consumer should call reuse-agents workflow"
    assert (
        "timeout-minutes" not in dispatch_job
    ), "Timeout must be delegated to reuse-agents -> reusable-70-agents"


def test_reusable_agents_jobs_have_timeouts():
    data = _load_workflow_yaml("reusable-70-agents.yml")
    jobs = data.get("jobs", {})
    missing_timeouts = [
        name
        for name, job in jobs.items()
        if isinstance(job, dict) and job.get("runs-on") and "timeout-minutes" not in job
    ]
    assert not missing_timeouts, f"Jobs missing timeout-minutes: {missing_timeouts}"
