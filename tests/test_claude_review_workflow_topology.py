import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"


def test_only_current_non_blocking_claude_review_lane_remains() -> None:
    legacy = WORKFLOWS / "claude-code-review.yml"
    current = WORKFLOWS / "maint-76-claude-code-review.yml"

    assert not legacy.exists()
    assert current.exists()

    text = current.read_text(encoding="utf-8")
    automatic_claude_reviewers = []
    for workflow in WORKFLOWS.glob("*.yml"):
        workflow_text = workflow.read_text(encoding="utf-8")
        if "anthropics/claude-code-action" in workflow_text and re.search(
            r"^  pull_request:\s*$", workflow_text, re.MULTILINE
        ):
            automatic_claude_reviewers.append(workflow.name)

    assert automatic_claude_reviewers == [current.name]
    assert "workflow_dispatch:" in text
    assert 'labels.includes("claude-review")' in text
    assert 'let shouldRun = "false"' in text
    assert "continue-on-error: true" in text
    assert "--max-turns 8" in text
    assert "Record review failure (non-blocking)" in text
