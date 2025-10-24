import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def get_default_marker():
    script = """
const path = require('path');
const { DEFAULT_MARKER } = require(path.resolve(process.cwd(), '.github/scripts/agents-guard.js'));
process.stdout.write(DEFAULT_MARKER);
"""

    completed = subprocess.run(
        ["node", "-e", script],
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        check=True,
    )
    return completed.stdout


DEFAULT_MARKER = get_default_marker()


def run_guard(
    files=None,
    labels=None,
    reviews=None,
    codeowners=None,
    protected=None,
    author=None,
    marker=None,
):
    payload = {
        "files": files or [],
        "labels": labels or [],
        "reviews": reviews or [],
        "codeownersContent": codeowners or "",
    }
    if protected is not None:
        payload["protectedPaths"] = protected
    if author is not None:
        payload["authorLogin"] = author
    if marker is not None:
        payload["marker"] = marker

    script = """
const fs = require('fs');
const path = require('path');
const input = JSON.parse(fs.readFileSync(0, 'utf-8'));
const { evaluateGuard } = require(path.resolve(process.cwd(), '.github/scripts/agents-guard.js'));
const result = evaluateGuard(input);
process.stdout.write(JSON.stringify(result));
"""

    completed = subprocess.run(
        ["node", "-e", script],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        check=True,
    )
    return json.loads(completed.stdout)


CODEOWNERS_SAMPLE = """
# Example CODEOWNERS entries
/.github/workflows/** @stranske
""".strip()


def test_deletion_blocks_with_comment():
    result = run_guard(
        files=[
            {
                "filename": ".github/workflows/agents-63-chatgpt-issue-sync.yml",
                "status": "removed",
            }
        ],
        codeowners=CODEOWNERS_SAMPLE,
    )

    assert result["blocked"] is True
    assert any("was deleted" in reason for reason in result["failureReasons"])
    assert "Health 45 Agents Guard" in result["summary"]
    assert result["commentBody"].startswith(DEFAULT_MARKER)


def test_custom_marker_propagates_to_comment():
    custom_marker = "<!-- custom-guard-marker -->"
    result = run_guard(
        files=[
            {
                "filename": ".github/workflows/agents-63-chatgpt-issue-sync.yml",
                "status": "removed",
            }
        ],
        codeowners=CODEOWNERS_SAMPLE,
        marker=custom_marker,
    )

    assert result["marker"] == custom_marker
    assert result["commentBody"].startswith(custom_marker)


def test_default_marker_added_once():
    result = run_guard(
        files=[
            {
                "filename": ".github/workflows/agents-63-chatgpt-issue-sync.yml",
                "status": "removed",
            }
        ],
        codeowners=CODEOWNERS_SAMPLE,
    )

    assert result["commentBody"].startswith(DEFAULT_MARKER)
    assert result["commentBody"].count(DEFAULT_MARKER) == 1


def test_rename_blocks_with_guidance():
    result = run_guard(
        files=[
            {
                "filename": ".github/workflows/agents-63-codex-issue-bridge.yml",
                "previous_filename": ".github/workflows/agents-63-codex-issue-bridge.yml",
                "status": "renamed",
            }
        ],
        codeowners=CODEOWNERS_SAMPLE,
    )

    assert result["blocked"] is True
    assert any("was renamed" in reason for reason in result["failureReasons"])


def test_modification_without_label_or_approval_requires_both():
    result = run_guard(
        files=[
            {
                "filename": ".github/workflows/agents-70-orchestrator.yml",
                "status": "modified",
            }
        ],
        labels=[],
        reviews=[],
        codeowners=CODEOWNERS_SAMPLE,
    )

    assert result["blocked"] is True
    assert "Missing `agents:allow-change` label." in result["failureReasons"]
    assert any("Request approval" in reason for reason in result["failureReasons"])


def test_label_without_codeowner_still_blocks():
    result = run_guard(
        files=[
            {
                "filename": ".github/workflows/agents-63-chatgpt-issue-sync.yml",
                "status": "modified",
            }
        ],
        labels=[{"name": "agents:allow-change"}],
        reviews=[],
        codeowners=CODEOWNERS_SAMPLE,
    )

    assert result["blocked"] is True
    assert "Missing `agents:allow-change` label." not in result["failureReasons"]
    assert any("Request approval" in reason for reason in result["failureReasons"])


def test_label_and_codeowner_approval_passes():
    result = run_guard(
        files=[
            {
                "filename": ".github/workflows/agents-63-chatgpt-issue-sync.yml",
                "status": "modified",
            }
        ],
        labels=[{"name": "agents:allow-change"}],
        reviews=[
            {
                "user": {"login": "stranske"},
                "state": "APPROVED",
            }
        ],
        codeowners=CODEOWNERS_SAMPLE,
    )

    assert result["blocked"] is False
    assert result["summary"] == "Health 45 Agents Guard passed."


def test_codeowner_author_counts_as_approval():
    result = run_guard(
        files=[
            {
                "filename": ".github/workflows/agents-63-chatgpt-issue-sync.yml",
                "status": "modified",
            }
        ],
        reviews=[],
        codeowners=CODEOWNERS_SAMPLE,
        author="stranske",
    )

    assert result["blocked"] is False
    assert result["hasCodeownerApproval"] is True
    assert result["authorIsCodeowner"] is True
    assert result["hasAllowLabel"] is False


def test_codeowner_review_without_label_passes():
    result = run_guard(
        files=[
            {
                "filename": ".github/workflows/agents-70-orchestrator.yml",
                "status": "modified",
            }
        ],
        labels=[],
        reviews=[
            {
                "user": {"login": "stranske"},
                "state": "APPROVED",
            }
        ],
        codeowners=CODEOWNERS_SAMPLE,
    )

    assert result["blocked"] is False
    assert result["hasCodeownerApproval"] is True
    assert result["hasAllowLabel"] is False


def test_codeowner_approval_without_label_passes():
    result = run_guard(
        files=[
            {
                "filename": ".github/workflows/agents-63-chatgpt-issue-sync.yml",
                "status": "modified",
            }
        ],
        labels=[],
        reviews=[
            {
                "user": {"login": "stranske"},
                "state": "APPROVED",
            }
        ],
        codeowners=CODEOWNERS_SAMPLE,
    )

    assert result["blocked"] is False
    assert result["hasCodeownerApproval"] is True
    assert result["hasAllowLabel"] is False


def test_unprotected_file_is_ignored():
    result = run_guard(
        files=[
            {
                "filename": ".github/workflows/agents-64-verify-agent-assignment.yml",
                "status": "modified",
            }
        ],
        labels=[],
        reviews=[],
        codeowners=CODEOWNERS_SAMPLE,
    )

    assert result["blocked"] is False
    assert result["failureReasons"] == []
