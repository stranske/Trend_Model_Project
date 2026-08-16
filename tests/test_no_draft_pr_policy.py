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


def test_current_operator_instructions_do_not_restore_retired_orchestrator() -> None:
    operator_docs = (
        "Agents.md",
        "CONTRIBUTING.md",
        ".github/workflows/README.md",
        "docs/AGENTS_POLICY.md",
        "docs/ci/AGENTS_POLICY.md",
        "docs/WORKFLOW_GUIDE.md",
        "docs/ci/WORKFLOWS.md",
        "docs/ci/WORKFLOW_SYSTEM.md",
        "docs/ops/codex-bootstrap-facts.md",
        "docs/agent-automation.md",
        "docs/LABELS.md",
        "docs/prompts/library.md",
        "docs/ci_reuse.md",
        "docs/SETUP_CHECKLIST.md",
    )

    retired_tokens = (
        "agents-70-orchestrator.yml",
        "reusable-16-agents.yml",
        "agents-63-issue-intake.yml",
        "bridge_draft_pr",
        "agent_pr_draft",
        "draft_pr",
    )
    stale = []
    for relative in operator_docs:
        text = (ROOT / relative).read_text(encoding="utf-8")
        matches = [token for token in retired_tokens if token in text]
        if matches:
            stale.append((relative, matches))

    assert not stale, f"retired consumer automation remains in: {stale}"

    codeowners = (ROOT / ".github/CODEOWNERS").read_text(encoding="utf-8")
    assert "agents-70-orchestrator.yml" not in codeowners
    assert "/.github/workflows/agents-issue-intake.yml" in codeowners
