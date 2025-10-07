from pathlib import Path

WORKFLOWS_DIR = Path(".github/workflows")


def test_agents_orchestrator_inputs_and_uses():
    wf = WORKFLOWS_DIR / "agents-70-orchestrator.yml"
    assert wf.exists(), "agents-70-orchestrator.yml must exist"
    text = wf.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text, "Orchestrator must allow manual dispatch"
    assert "params_json:" in text, "params_json input must be documented"
    assert (
        "Parse params_json" in text
    ), "Parsing step must exist to unpack consolidated payload"
    assert (
        "needs: parse" in text
    ), "Reusable workflow call must depend on the parse job"
    assert (
        "needs.parse.outputs.enable_readiness" in text
    ), "Parsed outputs must be threaded into the reusable workflow"
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
        "options_json",
    ]:
        assert f"{key}:" in text, f"Reusable agents workflow must expose input: {key}"
    for output in ["issue_numbers_json", "first_issue"]:
        assert (
            output in text
        ), f"Find Ready Issues step must emit {output} output for bootstrap consumers"


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
