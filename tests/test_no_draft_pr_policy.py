from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_retired_local_bootstrap_surfaces_stay_removed() -> None:
    retired_paths = (
        ".github/actions/codex-bootstrap-lite/action.yml",
        "scripts/verify_codex_bootstrap.py",
        "tests/scripts/test_verify_codex_bootstrap_timestamps.py",
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
