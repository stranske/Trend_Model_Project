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


def test_agents_consumer_uses_params_json():
    wf = WORKFLOWS_DIR / "agents-consumer.yml"
    assert wf.exists(), "agents-consumer.yml must exist"
    text = wf.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text, "Consumer must allow manual dispatch"
    assert "params_json:" in text, "Consumer must consolidate inputs into params_json"
    # Ensure legacy discrete inputs are not reintroduced at the dispatch boundary
    forbidden = [
        "enable_readiness:",
        "readiness_agents:",
        "require_all:",
        "enable_preflight:",
        "enable_verify_issue:",
        "enable_watchdog:",
        "draft_pr:",
    ]
    header_section = text.split("jobs:", 1)[0]
    for key in forbidden:
        assert (
            key not in header_section
        ), f"Consumer dispatch header should not expose discrete input {key}"
    assert (
        "./.github/workflows/reuse-agents.yml" in text
    ), "Consumer must call reuse-agents.yml"


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
