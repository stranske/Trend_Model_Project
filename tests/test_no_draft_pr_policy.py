import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_retired_local_bootstrap_surfaces_stay_removed() -> None:
    retired_paths = (
        ".github/actions/codex-bootstrap-lite/action.yml",
        "scripts/verify_codex_bootstrap.py",
        "tests/scripts/test_verify_codex_bootstrap_timestamps.py",
        "tools/simulate_codex_bootstrap.py",
        "tests/test_simulate_codex_bootstrap.py",
        "docs/agent_codex_troubleshooting.md",
        "docs/codex-simulation.md",
        "docs/codex_bootstrap_verification.md",
    )

    assert not [path for path in retired_paths if (ROOT / path).exists()]


def test_repo_instructions_require_ready_for_review_pull_requests() -> None:
    agents = (ROOT / "Agents.md").read_text(encoding="utf-8")
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    facts = (ROOT / "docs/ops/codex-bootstrap-facts.md").read_text(encoding="utf-8")

    assert "verify `isDraft=false` before handoff" in agents
    assert "Open a draft PR early" not in contributing
    assert "do not use draft state" in contributing
    assert "Automation-created pull requests are opened ready for review" in facts


def test_active_intake_has_no_draft_control() -> None:
    intake = (ROOT / ".github/workflows/agents-issue-intake.yml").read_text(encoding="utf-8")

    assert "bridge_draft_pr" not in intake
    assert "agent_pr_draft" not in intake


def test_documented_agent_entrypoints_match_workflow_topology() -> None:
    workflows = ROOT / ".github" / "workflows"
    dispatchable = (
        "agents-auto-pilot.yml",
        "agents-71-codex-belt-dispatcher.yml",
        "agents-72-codex-belt-worker-dispatch.yml",
    )
    for filename in dispatchable:
        text = (workflows / filename).read_text(encoding="utf-8")
        assert "workflow_dispatch:" in text, filename

    callable_only = (
        "agents-72-codex-belt-worker.yml",
        "agents-73-codex-belt-conveyor.yml",
    )
    for filename in callable_only:
        text = (workflows / filename).read_text(encoding="utf-8")
        assert "workflow_call:" in text, filename
        assert "workflow_dispatch:" not in text, filename

    wrapper = (workflows / "agents-72-codex-belt-worker-dispatch.yml").read_text(encoding="utf-8")
    auto_pilot = (workflows / "agents-auto-pilot.yml").read_text(encoding="utf-8")
    assert "uses: ./.github/workflows/agents-72-codex-belt-worker.yml" in wrapper
    assert "workflow_id: 'agents-71-codex-belt-dispatcher.yml'" in auto_pilot
    assert "workflow_id: 'agents-72-codex-belt-worker-dispatch.yml'" in auto_pilot

    quick_start = (workflows / "README.md").read_text(encoding="utf-8")
    primary_table = quick_start.split("## Primary Entry Points", maxsplit=1)[1].split(
        "The old consumer orchestrator", maxsplit=1
    )[0]
    assert "`agents-auto-pilot.yml`" in primary_table
    assert "`agents-72-codex-belt-worker-dispatch.yml`" in primary_table
    assert "`agents-72-codex-belt-worker.yml`" not in primary_table
    assert "`agents-73-codex-belt-conveyor.yml`" not in primary_table

    protected_policy = (ROOT / "docs/AGENTS_POLICY.md").read_text(encoding="utf-8")
    assert "`.github/workflows/agents-auto-pilot.yml`" in protected_policy


def test_current_operator_instructions_do_not_restore_retired_orchestrator() -> None:
    operator_docs = (
        "Agents.md",
        "CONTRIBUTING.md",
        "WORKFLOW_USER_GUIDE.md",
        ".github/workflows/README.md",
        "docs/AGENTS_POLICY.md",
        "docs/ci/AGENTS_POLICY.md",
        "docs/WORKFLOW_GUIDE.md",
        "docs/ci/WORKFLOWS.md",
        "docs/ci/WORKFLOW_SYSTEM.md",
        "docs/ci/ISSUE_FORMAT_GUIDE.md",
        "docs/ops/codex-bootstrap-facts.md",
        "docs/ops/template-setup.md",
        "docs/agent-automation.md",
        "docs/workflow-chatgpt-issue-sync.md",
        "docs/LABELS.md",
        "docs/prompts/library.md",
        "docs/ci_reuse.md",
        "docs/SETUP_CHECKLIST.md",
        ".github/workflows/agents-auto-pilot.yml",
    )

    retired_tokens = (
        "agents-70-orchestrator.yml",
        "reusable-16-agents.yml",
        "agents-63-issue-intake.yml",
        "bridge_draft_pr",
        "agent_pr_draft",
        "draft_pr",
        "agents-63",
        "Agents 63",
        "agents-pr-meta.yml",
        "agents-orchestrator.yml",
        "agents-moderate-connector.yml",
        "agents-keepalive-branch-sync.yml",
        "agents-keepalive-dispatch-handler.yml",
        "agents-debug-issue-event.yml",
        "PR Meta",
        "pr_meta_comment",
        "allow_replay",
        "raw.githubusercontent.com/stranske/Workflows/v1",
        "@v1",
        "agents-keepalive-loop.yml",
        "agents-pr-meta-v4.yml",
    )
    affirmative_draft_instruction = re.compile(
        r"(?<!non-)(?<!no )(?<!not )\bdraft\s+(?:PR|pull request)\b",
        re.IGNORECASE,
    )
    stale = []
    for relative in operator_docs:
        text = (ROOT / relative).read_text(encoding="utf-8")
        matches = [token for token in retired_tokens if token in text]
        if affirmative_draft_instruction.search(text):
            matches.append("affirmative draft-PR instruction")
        if matches:
            stale.append((relative, matches))

    assert not stale, f"retired consumer automation remains in: {stale}"

    retired_dispatch_targets = (
        "agents-keepalive-loop.yml",
        "agents-pr-meta-v4.yml",
    )
    stale_workflows = []
    for path in sorted((ROOT / ".github/workflows").glob("*.y*ml")):
        text = path.read_text(encoding="utf-8")
        matches = [token for token in retired_dispatch_targets if token in text]
        if matches:
            stale_workflows.append((path.relative_to(ROOT).as_posix(), matches))

    assert not stale_workflows, f"retired dispatch targets remain in: {stale_workflows}"

    checklist = (ROOT / "docs/SETUP_CHECKLIST.md").read_text(encoding="utf-8")
    assert "Keepalive Sweep re-enters the Agents 81 evaluation" in checklist
    assert checklist.count("both `agent:codex` and `agents:keepalive` labels") == 2

    codeowners = (ROOT / ".github/CODEOWNERS").read_text(encoding="utf-8")
    assert "agents-70-orchestrator.yml" not in codeowners
    assert "/.github/workflows/agents-issue-intake.yml" in codeowners
