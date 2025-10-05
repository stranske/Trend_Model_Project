from pathlib import Path

WORKFLOWS_DIR = Path(".github/workflows")


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
    ]:
        assert f"{key}:" in text, f"Reusable agents workflow must expose input: {key}"


def test_legacy_agent_workflows_removed():
    present = {p.name for p in WORKFLOWS_DIR.glob("agents-*.yml")}
    forbidden = {
        "agents-40-consumer.yml",
        "agents-41-assign-and-watch.yml",
        "agents-41-assign.yml",
        "agents-42-watchdog.yml",
        "agents-43-codex-issue-bridge.yml",
        "agents-44-copilot-readiness.yml",
        "agents-45-verify-codex-bootstrap-matrix.yml",
    }
    assert not (
        present & forbidden
    ), f"Legacy agent workflows still present: {present & forbidden}"
