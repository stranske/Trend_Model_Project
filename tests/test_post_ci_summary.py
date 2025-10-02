from __future__ import annotations

import json

import pytest

from tools.post_ci_summary import build_summary_comment


@pytest.fixture()
def sample_runs() -> list[dict[str, object]]:
    return [
        {
            "key": "ci",
            "displayName": "CI",
            "present": True,
            "id": 101,
            "run_attempt": 1,
            "html_url": "https://example.test/ci/101",
            "jobs": [
                {
                    "name": "main / tests (3.9)",
                    "conclusion": "success",
                    "html_url": "https://example.test/ci/101/tests",
                },
                {
                    "name": "main / style",
                    "conclusion": "success",
                    "html_url": "https://example.test/ci/101/style",
                },
                {
                    "name": "workflow / automation-tests",
                    "status": "queued",
                    "html_url": "https://example.test/ci/101/workflow",
                },
                {
                    "name": "gate / all-required-green",
                    "conclusion": "success",
                    "html_url": "https://example.test/ci/101/gate",
                },
            ],
        },
        {
            "key": "docker",
            "displayName": "Docker",
            "present": True,
            "id": 202,
            "run_attempt": 2,
            "conclusion": "failure",
            "html_url": "https://example.test/docker/202",
            "jobs": [
                {
                    "name": "docker build",
                    "conclusion": "failure",
                    "html_url": "https://example.test/docker/202/build",
                }
            ],
        },
    ]


def test_build_summary_comment_renders_expected_sections(
    sample_runs: list[dict[str, object]],
) -> None:
    coverage_stats = {
        "avg_latest": 91.23,
        "avg_delta": -0.77,
        "worst_latest": 83.11,
        "worst_delta": 1.02,
        "history_len": 12,
    }
    coverage_section = "Extra metrics here"

    body = build_summary_comment(
        runs=sample_runs,
        head_sha="abc123",
        coverage_stats=coverage_stats,
        coverage_section=coverage_section,
        required_groups_env=json.dumps(
            [
                {"label": "CI tests", "patterns": [r"^main / tests"]},
                {"label": "CI gate", "patterns": [r"^gate /"]},
            ]
        ),
    )

    assert "<!-- post-ci-summary:do-not-edit -->" in body
    assert "### Automated Status Summary" in body
    assert "**Head SHA:** abc123" in body
    assert "**Latest Runs:**" in body
    assert "CI tests: ✅ success" in body
    assert "CI gate: ✅ success" in body
    assert "Docker: ❌ failure" in body
    assert "| CI / main / tests (3.9) | ✅ success |" in body
    assert "| **Docker / docker build** | ❌ failure |" in body
    # Coverage lines should render with percentages and deltas
    assert "Coverage (jobs): 91.23%" in body
    assert "Coverage (worst job): 83.11%" in body
    assert "Δ -0.77 pp" in body
    assert "Δ +1.02 pp" in body
    assert coverage_section in body


def test_build_summary_comment_handles_missing_runs_and_defaults() -> None:
    body = build_summary_comment(
        runs=[{"key": "ci", "displayName": "CI", "present": False}],
        head_sha=None,
        coverage_stats=None,
        coverage_section=None,
        required_groups_env=None,
    )

    assert "CI: ⏳ pending" in body
    assert "Docker: ⏳ pending" in body
    assert "_Updated automatically; will refresh" in body
    # When no jobs exist the fallback table entry is rendered
    assert "_(no jobs reported)_" in body
