from pathlib import Path

WORKFLOWS_DIR = Path(".github/workflows")


def _load_yaml_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_assign_workflow_present_and_minimal():
    wf = WORKFLOWS_DIR / "agents-41-assign.yml"
    assert wf.exists(), "agents-41-assign.yml must exist (agents consolidation guard)"
    text = _load_yaml_text(wf)
    assert (
        "on:" in text and "issues:" in text
    ), "agents-41-assign.yml must listen to issue events"
    assert (
        "pull_request_target:" in text
    ), "agents-41-assign.yml must handle PR labeling events"
    assert (
        "codex-bootstrap-lite" in text
    ), "Codex bootstrap composite action must be invoked"
    assert "@codex start" in text, "Trigger command for Codex must be present"
    assert "agents-42-watchdog.yml" in text, "Watchdog dispatch must remain wired"


def test_agent_watchdog_timeout_and_messages():
    wf = WORKFLOWS_DIR / "agents-42-watchdog.yml"
    assert wf.exists(), "agents-42-watchdog.yml must exist (agents consolidation guard)"
    text = _load_yaml_text(wf)
    assert "timeout_minutes" in text, "Watchdog input timeout_minutes missing"
    assert "✅ Agent Watchdog" in text, "Success message prefix missing"
    assert "⚠️ Agent Watchdog" in text, "Timeout message prefix missing"


def test_no_active_legacy_agent_workflows():
    active = {p.name for p in WORKFLOWS_DIR.glob("*.yml")}
    legacy_names = {"agents-consumer.yml", "reuse-agents.yml"}
    assert active.isdisjoint(
        legacy_names
    ), f"Legacy agent workflows still active: {active & legacy_names}"


def test_agents_consumer_and_reusable_present():
    consumer = WORKFLOWS_DIR / "agents-40-consumer.yml"
    reusable = WORKFLOWS_DIR / "reusable-90-agents.yml"
    assert consumer.exists(), "agents-40-consumer.yml must exist"
    assert reusable.exists(), "reusable-90-agents.yml must exist"
