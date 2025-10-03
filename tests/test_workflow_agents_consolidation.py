from pathlib import Path

WORKFLOWS_DIR = Path(".github/workflows")


def _load_yaml_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_unified_agents_workflow_structure():
    wf = WORKFLOWS_DIR / "agents-41-assign-and-watch.yml"
    assert wf.exists(), "agents-41-assign-and-watch.yml must exist (unified orchestrator guard)"
    text = _load_yaml_text(wf)
    assert "workflow_dispatch:" in text, "Unified workflow must expose workflow_dispatch trigger"
    assert "schedule:" in text, "Unified workflow must include scheduled watchdog sweep"
    assert "reusable-90-agents.yml" in text, "Unified workflow must reuse reusable-90-agents for readiness checks"
    assert "codex-bootstrap-lite" in text, "Unified workflow must invoke codex bootstrap composite"
    assert "watchdog_sweep" in text, "Unified workflow must define the watchdog sweep job"
    assert "✅ Agent Watchdog" in text, "Unified workflow must preserve success watchdog messaging"
    assert "🚨 Watchdog escalation" in text, "Unified workflow must emit escalation prefix for stale issues"


def test_assign_wrapper_forwards_events():
    wf = WORKFLOWS_DIR / "agents-41-assign.yml"
    assert wf.exists(), "agents-41-assign.yml wrapper must remain present"
    text = _load_yaml_text(wf)
    assert "issues:" in text, "Wrapper must still listen to issue events"
    assert "createWorkflowDispatch" in text, "Wrapper must forward via workflow dispatch"
    assert "workflow_id: 'agents-41-assign-and-watch.yml'" in text, "Wrapper must delegate to unified workflow"
    assert "event_payload" in text, "Wrapper must forward raw event payload"


def test_watchdog_wrapper_dispatches_mode():
    wf = WORKFLOWS_DIR / "agents-42-watchdog.yml"
    assert wf.exists(), "agents-42-watchdog.yml wrapper must remain present"
    text = _load_yaml_text(wf)
    assert "workflow_dispatch:" in text, "Watchdog wrapper must keep manual trigger"
    assert "mode: 'watch'" in text, "Watchdog wrapper must call unified workflow with mode=watch"
    assert "event_payload" in text, "Watchdog wrapper must forward payload context"


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
    consumer_text = _load_yaml_text(consumer)
    reusable_text = _load_yaml_text(reusable)
    assert (
        "uses: ./.github/workflows/reusable-90-agents.yml" in consumer_text
    ), "agents-40-consumer.yml must call reusable-90-agents.yml"
    assert (
        "workflow_call:" in reusable_text
    ), "reusable-90-agents.yml must expose a workflow_call trigger"
    assert (
        "fromJSON(format('[{0}]', steps.ready.outputs.issue_numbers))[0]" in reusable_text
    ), "Bootstrap expression must parse issue numbers via format()"
    assert (
        "'[' + steps.ready.outputs.issue_numbers + ']'" not in reusable_text
    ), "Legacy string concatenation pattern must not remain in reusable-90-agents.yml"
