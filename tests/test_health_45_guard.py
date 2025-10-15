import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_guard(files=None, labels=None, reviews=None, codeowners=None, protected=None):
    payload = {
        "files": files or [],
        "labels": labels or [],
        "reviews": reviews or [],
        "codeownersContent": codeowners or "",
    }
    if protected is not None:
        payload["protectedPaths"] = protected

    script = """
const fs = require('fs');
const path = require('path');
const input = JSON.parse(fs.readFileSync(0, 'utf-8'));
const { evaluateGuard } = require(path.resolve(process.cwd(), '.github/scripts/health-45-guard.js'));
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
    assert result["commentBody"].startswith("<!-- health-45-agents-guard -->")


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
