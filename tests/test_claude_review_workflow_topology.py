from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"


def test_only_current_non_blocking_claude_review_lane_remains() -> None:
    legacy = WORKFLOWS / "claude-code-review.yml"
    current = WORKFLOWS / "maint-76-claude-code-review.yml"

    assert not legacy.exists()
    assert current.exists()

    text = current.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "continue-on-error: true" in text
    assert "--max-turns 8" in text
    assert "Record review failure (non-blocking)" in text
