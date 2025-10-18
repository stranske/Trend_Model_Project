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
        "bootstrap_issues_label not provided; defaulting to agent:codex." in text
    ), "Orchestrator must log when it falls back to the default Codex label"
    assert (
        "const bootstrapLabelCandidate = toString" in text
    ), "Orchestrator must sanitise the bootstrap label input before dispatch"
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


def test_keepalive_job_defined_once():
    data = _load_workflow_yaml("reusable-16-agents.yml")
    jobs = data.get("jobs", {})
    keepalive_jobs = [
        (name, job.get("name"))
        for name, job in jobs.items()
        if isinstance(job, dict)
        and isinstance(job.get("name"), str)
        and "Codex Keepalive" in job.get("name")
    ]
    assert keepalive_jobs == [
        ("keepalive", "Codex Keepalive Sweep")
    ], "Reusable workflow must expose a single Codex keepalive job"


def test_bootstrap_requires_single_label():
    text = (WORKFLOWS_DIR / "reusable-16-agents.yml").read_text(encoding="utf-8")
    assert (
        "bootstrap_issues_label not provided; defaulting to agent:codex." in text
    ), "Bootstrap step must record when it falls back to the default label"
    assert (
        "bootstrap_issues_label input must define exactly one label" in text
    ), "Bootstrap step must prevent sweeping multiple labels"


def test_bootstrap_filters_by_requested_label():
    text = (WORKFLOWS_DIR / "reusable-16-agents.yml").read_text(encoding="utf-8")
    assert (
        "labels: label" in text
    ), "Bootstrap GitHub API call must request only the configured label"
    assert (
        "missing required label ${label}" in text
    ), "Bootstrap script must skip issues that do not carry the requested label"


def test_bootstrap_uses_paginated_issue_scan():
    text = (WORKFLOWS_DIR / "reusable-16-agents.yml").read_text(encoding="utf-8")
    assert (
        "github.paginate.iterator" in text
    ), "Bootstrap must paginate issue scanning to avoid truncation"
    assert (
        "Evaluated issues:" in text
    ), "Bootstrap summary should report how many issues were inspected"


def test_bootstrap_summary_includes_scope_and_counts():
    text = (WORKFLOWS_DIR / "reusable-16-agents.yml").read_text(encoding="utf-8")
    assert (
        "Bootstrap label: **" in text
    ), "Bootstrap run summary should surface the resolved label scope"
    assert "Skipped issues" in text, "Bootstrap summary must document skipped issues"
    assert (
        "Accepted issues:" in text
    ), "Bootstrap summary must include accepted issue counts"
    assert (
        "Skipped issues:" in text
    ), "Bootstrap summary must include skipped issue counts"
    assert (
        "https://github.com/" in text
    ), "Bootstrap summary should link directly to accepted issues"


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


def test_agents_orchestrator_schedule_preserved():
    data = _load_workflow_yaml("agents-70-orchestrator.yml")

    on_section = _workflow_on_section(data)
    schedule = on_section.get("schedule") or []
    assert schedule, "Orchestrator schedule must remain defined"

    cron_entries = [
        entry.get("cron")
        for entry in schedule
        if isinstance(entry, dict) and "cron" in entry
    ]
    assert cron_entries == [
        "*/20 * * * *"
    ], "Orchestrator schedule must stay on the 20-minute cadence"


def test_bootstrap_step_defaults_label_when_missing():
    text = (WORKFLOWS_DIR / "reusable-16-agents.yml").read_text(encoding="utf-8")
    assert (
        "const fallbackLabel = 'agent:codex'" in text
    ), "Bootstrap logic must define agent:codex as the fallback label"
    assert (
        "bootstrap_issues_label not provided; defaulting to" in text
    ), "Bootstrap step must record when it falls back to the default label"


def test_agents_consumer_workflow_removed():
    path = WORKFLOWS_DIR / "agents-62-consumer.yml"
    assert not path.exists(), "Retired Agents 62 consumer wrapper must remain absent"


def test_agent_task_template_auto_labels_codex():
    template = Path(".github/ISSUE_TEMPLATE/agent_task.yml")
    assert template.exists(), "Agent task issue template must exist"
    data = yaml.safe_load(template.read_text(encoding="utf-8"))
    labels = set(data.get("labels") or [])
    assert {"agents", "agent:codex"}.issubset(
        labels
    ), "Agent task template must auto-apply agents + agent:codex labels"


def test_codex_issue_bridge_triggers_on_agent_label():
    data = _load_workflow_yaml("agents-63-codex-issue-bridge.yml")
    triggers = _workflow_on_section(data)
    assert "issues" in triggers, "Codex issue bridge must listen for issue events"
    issue_trigger = triggers["issues"] or {}
    types = set(issue_trigger.get("types") or [])
    assert {"opened", "labeled", "reopened"}.issubset(
        types
    ), "Issue bridge must react to issue label lifecycle events"
    text = (WORKFLOWS_DIR / "agents-63-codex-issue-bridge.yml").read_text(
        encoding="utf-8"
    )
    assert (
        "agent:codex" in text
    ), "Issue bridge must guard on the agent:codex label to trigger hand-off"


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


def test_keepalive_summary_reports_scope_and_activity():
    text = (WORKFLOWS_DIR / "reusable-16-agents.yml").read_text(encoding="utf-8")
    assert "Target labels:" in text, "Keepalive summary should list the label scope"
    assert (
        "Agent logins:" in text
    ), "Keepalive summary should surface the Codex logins under consideration"
    assert (
        "No unattended Codex tasks detected." in text or "keepalive posted" in text
    ), "Keepalive summary must describe whether any PRs required intervention"
    assert (
        "Triggered keepalive comments" in text
    ), "Keepalive summary should wrap triggered comment list in a collapsible section"
    assert (
        "Triggered keepalive count:" in text
    ), "Keepalive summary should record how many follow-up comments were sent"


def test_keepalive_job_runs_after_failures():
    data = _load_workflow_yaml("reusable-16-agents.yml")
    jobs = data.get("jobs", {})
    keepalive = jobs.get("keepalive")
    assert keepalive, "Reusable workflow must define keepalive job"
    assert (
        keepalive.get("if") == "${{ always() && inputs.enable_keepalive == 'true' }}"
    ), "Keepalive job must run even if earlier jobs fail while respecting enable_keepalive flag"


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
