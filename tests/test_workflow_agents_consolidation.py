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
    expected_inputs = {"params_json"}
    for key in expected_inputs:
        assert f"{key}:" in text, f"Missing workflow_dispatch input: {key}"
    assert (
        "github.event.inputs.params_json" in text
    ), "params_json must be read from workflow_dispatch inputs"
    assert "PARAMS_JSON" in text, "Resolve step must pass params_json via env"
    assert "JSON.parse" in text, "params_json must be parsed as JSON"
    assert "options_json" in text, "options_json output must remain available"
    assert (
        "enable_bootstrap:" in text
    ), "Orchestrator must forward enable_bootstrap flag"
    assert (
        "bootstrap_issues_label:" in text
    ), "Orchestrator must forward bootstrap label configuration"
    assert (
        "./.github/workflows/reusable-16-agents.yml" in text
    ), "Orchestrator must call the reusable agents workflow"


def test_reusable_agents_workflow_structure():
    reusable = WORKFLOWS_DIR / "reusable-16-agents.yml"
    assert reusable.exists(), "reusable-16-agents.yml must exist"
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


def test_agent_watchdog_workflow_absent():
    legacy_watchdog = WORKFLOWS_DIR / "agent-watchdog.yml"
    assert (
        not legacy_watchdog.exists()
    ), "Standalone agent-watchdog workflow must remain deleted"


def test_codex_issue_bridge_present():
    bridge = WORKFLOWS_DIR / "agents-63-codex-issue-bridge.yml"
    assert (
        bridge.exists()
    ), "agents-63-codex-issue-bridge.yml must exist after Codex bridge restoration"


def test_keepalive_job_present():
    reusable = WORKFLOWS_DIR / "reusable-16-agents.yml"
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
        "Job timeouts live inside reusable-16-agents.yml" in text
    ), "Orchestrator workflow should document where the timeout is enforced"


def test_agents_consumer_workflow_removed():
    path = WORKFLOWS_DIR / "agents-62-consumer.yml"
    assert not path.exists(), "Legacy Agents 62 consumer wrapper should be removed"


def test_reusable_agents_jobs_have_timeouts():
    data = _load_workflow_yaml("reusable-16-agents.yml")
    jobs = data.get("jobs", {})
    missing_timeouts = [
        name
        for name, job in jobs.items()
        if isinstance(job, dict) and job.get("runs-on") and "timeout-minutes" not in job
    ]
    assert not missing_timeouts, f"Jobs missing timeout-minutes: {missing_timeouts}"


def test_reusable_watchdog_job_gated_by_flag():
    data = _load_workflow_yaml("reusable-16-agents.yml")
    jobs = data.get("jobs", {})
    watchdog = jobs.get("watchdog")
    assert watchdog, "Reusable workflow must expose watchdog job"
    assert (
        watchdog.get("if") == "inputs.enable_watchdog == 'true'"
    ), "Watchdog job must respect enable_watchdog flag"
    assert (
        watchdog.get("timeout-minutes") == 20
    ), "Watchdog job should retain the expected timeout"
    steps = watchdog.get("steps") or []
    assert any(
        isinstance(step, dict) and step.get("uses") == "actions/checkout@v4"
        for step in steps
    ), "Watchdog job must continue performing basic repo checks"


def test_orchestrator_forwards_enable_watchdog_flag():
    data = _load_workflow_yaml("agents-70-orchestrator.yml")
    jobs = data.get("jobs", {})
    orchestrate = jobs.get("orchestrate")
    assert orchestrate, "Orchestrator workflow must dispatch reusable agents job"
    with_section = orchestrate.get("with") or {}
    assert (
        with_section.get("enable_watchdog")
        == "${{ needs.resolve-params.outputs.enable_watchdog }}"
    ), "Orchestrator must forward enable_watchdog to the reusable workflow"
