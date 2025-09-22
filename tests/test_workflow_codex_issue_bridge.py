import re
from pathlib import Path

WORKFLOW_PATH = Path(".github/workflows/codex-issue-bridge.yml")


def test_codex_issue_bridge_workflow_exists():
    assert (
        WORKFLOW_PATH.exists()
    ), "Codex Issue Bridge workflow missing; restore codex-issue-bridge.yml"


def test_codex_issue_bridge_core_triggers_and_steps():
    content = WORKFLOW_PATH.read_text(encoding="utf-8")

    # Required top-level name
    assert (
        "name: Codex Issue Bridge" in content
    ), "Workflow name must be 'Codex Issue Bridge'"

    # Required triggers
    assert re.search(
        r"on:\s*\n\s*issues:", content
    ), "Workflow must trigger on issues events"
    assert (
        "workflow_dispatch:" in content
    ), "Workflow must support manual workflow_dispatch"

    # Required job key
    assert re.search(
        r"jobs:\s*\n\s*bridge:", content
    ), "Workflow must define 'bridge' job"

    # Critical steps keywords we rely on for automation integration
    required_markers = [
        "Resolve issue number",
        "Get default branch",
        "Try local composite Codex bootstrap (lite)",
        "Create branch and bootstrap file",
        "Open or reuse PR",
        "Post Codex command",
    ]
    for marker in required_markers:
        assert marker in content, f"Missing required step marker: {marker}"

    # Ensure @codex start command present (engagement trigger)
    assert (
        "@codex start" in content
    ), "Codex start command not found in workflow (required for engagement)."

    # Sanity check: concurrency group contains 'codex-issue-'
    assert (
        "concurrency:" in content and "codex-issue-" in content
    ), "Concurrency group for codex issues missing."

    # Permissions block must allow required write scopes
    for perm in ("contents: write", "pull-requests: write", "issues: write"):
        assert perm in content, f"Required permission missing: {perm}"
