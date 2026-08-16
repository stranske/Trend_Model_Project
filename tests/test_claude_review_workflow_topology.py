from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"


def _has_automatic_pull_request_trigger(workflow_text: str) -> bool:
    document = yaml.load(workflow_text, Loader=yaml.BaseLoader)
    triggers = document.get("on", {}) if isinstance(document, dict) else {}
    if isinstance(triggers, dict):
        trigger_names = set(triggers)
    elif isinstance(triggers, list):
        trigger_names = set(triggers)
    else:
        trigger_names = {triggers}
    return bool({"pull_request", "pull_request_target"} & trigger_names)


def test_only_current_non_blocking_claude_review_lane_remains() -> None:
    legacy = WORKFLOWS / "claude-code-review.yml"
    current = WORKFLOWS / "maint-76-claude-code-review.yml"

    assert not legacy.exists()
    assert current.exists()

    text = current.read_text(encoding="utf-8")
    automatic_claude_reviewers = []
    workflow_files = [*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")]
    for workflow in sorted(workflow_files):
        workflow_text = workflow.read_text(encoding="utf-8")
        if (
            "anthropics/claude-code-action" in workflow_text
            and _has_automatic_pull_request_trigger(workflow_text)
        ):
            automatic_claude_reviewers.append(workflow.name)

    assert automatic_claude_reviewers == [current.name]
    assert "workflow_dispatch:" in text
    assert 'labels.includes("claude-review")' in text
    assert 'let shouldRun = "false"' in text
    assert "continue-on-error: true" in text
    assert "--max-turns 8" in text
    assert "Record review failure (non-blocking)" in text
