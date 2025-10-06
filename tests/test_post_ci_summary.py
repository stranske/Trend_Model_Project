from __future__ import annotations

import json

from typing import Sequence

import pytest

from tools import post_ci_summary
from tools.post_ci_summary import (
    DEFAULT_REQUIRED_JOB_GROUPS,
    _load_required_groups,
    build_summary_comment,
)


@pytest.fixture()
def sample_runs() -> list[dict[str, object]]:
    return [
        {
            "key": "ci",
            "displayName": "CI",
            "present": True,
            "id": 101,
            "run_attempt": 1,
            "conclusion": "success",
            "html_url": "https://example.test/ci/101",
            "jobs": [
                {
                    "name": "ci / python",
                    "conclusion": "success",
                    "html_url": "https://example.test/ci/101/python",
                }
            ],
        },
        {
            "key": "docker",
            "displayName": "Docker",
            "present": True,
            "id": 202,
            "run_attempt": 2,
            "conclusion": "failure",
            "status": "completed",
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
                {"label": "CI python", "patterns": [r"^ci / python"]},
                {"label": "Docker", "patterns": [r"^docker "]},
            ]
        ),
    )

    assert "<!-- post-ci-summary:do-not-edit -->" in body
    assert "### Automated Status Summary" in body
    assert "**Head SHA:** abc123" in body
    assert (
        "**Latest Runs:** ✅ success — [CI (#101)](https://example.test/ci/101)" in body
    )
    assert (
        "· ❌ failure — [Docker (#202 (attempt 2))](https://example.test/docker/202)"
        in body
    )
    assert "CI python: ✅ success" in body
    assert "Docker: ❌ failure" in body
    assert "| CI / ci / python | ✅ success |" in body
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
    assert "**Latest Runs:** ⏳ pending — CI" in body
    assert "Docker: ⏳ pending" in body
    assert "_Updated automatically; will refresh" in body
    # When no jobs exist the fallback table entry is rendered
    assert "_(no jobs reported)_" in body


def test_job_table_prioritises_failing_and_pending_jobs(sample_runs):
    flaky_job = {
        "name": "main / flaky-suite",
        "conclusion": "failure",
        "html_url": "https://example.test/ci/101/flaky",
    }
    pending_job = {
        "name": "main / docs",
        "status": "in_progress",
        "html_url": "https://example.test/ci/101/docs",
    }
    skipped_job = {
        "name": "main / optional",
        "conclusion": "skipped",
        "html_url": "https://example.test/ci/101/optional",
    }

    # Inject additional jobs to exercise ordering logic.
    sample_runs[0]["jobs"].extend([flaky_job, pending_job, skipped_job])

    body = build_summary_comment(
        runs=sample_runs,
        head_sha="abc123",
        coverage_stats=None,
        coverage_section=None,
        required_groups_env=None,
    )

    table_lines = [
        line
        for line in body.splitlines()
        if line.startswith("| ")
        and "Workflow / Job" not in line
        and "------" not in line
    ]

    docker_index = next(
        (i for i, line in enumerate(table_lines) if "docker build" in line), None
    )
    flaky_index = next(
        (i for i, line in enumerate(table_lines) if "flaky-suite" in line), None
    )
    docs_index = next(
        (i for i, line in enumerate(table_lines) if "main / docs" in line), None
    )
    optional_index = next(
        (i for i, line in enumerate(table_lines) if "main / optional" in line), None
    )

    assert docker_index is not None, "'docker build' job not found in table_lines"
    assert flaky_index is not None, "'flaky-suite' job not found in table_lines"
    assert docs_index is not None, "'main / docs' job not found in table_lines"
    assert optional_index is not None, "'main / optional' job not found in table_lines"
    assert "❌ failure" in table_lines[docker_index]
    assert "❌ failure" in table_lines[flaky_index]
    assert "⏳ in progress" in table_lines[docs_index]
    assert "⏭️ skipped" in table_lines[optional_index]

    assert max(flaky_index, docker_index) < optional_index
    assert docs_index < optional_index


def test_coverage_section_handles_snippet_without_stats() -> None:
    snippet = "\nCoverage snippet from artifact.\n"

    body = build_summary_comment(
        runs=[],
        head_sha=None,
        coverage_stats=None,
        coverage_section=snippet,
        required_groups_env=None,
    )

    assert "### Coverage Overview" in body
    assert "Coverage snippet from artifact." in body
    assert body.count("### Coverage Overview") == 1


def test_latest_runs_handles_partial_metadata() -> None:
    runs = [
        {
            "key": "ci",
            "present": True,
            "id": None,
            "conclusion": None,
            "status": "in_progress",
            "html_url": None,
        },
        {
            "key": "docker",
            "display_name": "",
            "present": True,
            "id": 42,
            "run_attempt": None,
            "status": "queued",
            "html_url": "https://example.test/docker/42",
        },
        {
            "key": "lint",
            "present": False,
        },
    ]

    body = build_summary_comment(
        runs=runs,
        head_sha=None,
        coverage_stats=None,
        coverage_section=None,
        required_groups_env=None,
    )

    latest_line = next(
        (line for line in body.splitlines() if line.startswith("**Latest Runs:**")),
        "",
    )

    assert "⏳ in progress — ci" in latest_line.lower()
    assert "⏳ queued — [docker (#42)](https://example.test/docker/42)" in latest_line
    assert "⏳ pending — lint" in latest_line


def test_load_required_groups_handles_invalid_inputs() -> None:
    # Invalid JSON should fall back to defaults rather than raising.
    assert _load_required_groups("{invalid json}") == DEFAULT_REQUIRED_JOB_GROUPS

    # Non-list payloads also fall back to defaults.
    assert (
        _load_required_groups(json.dumps({"label": "ignored"}))
        == DEFAULT_REQUIRED_JOB_GROUPS
    )


def test_load_required_groups_filters_incomplete_entries() -> None:
    custom: Sequence[dict[str, object]] = [
        {"label": "Lint", "patterns": [r"^lint /"]},
        {"label": "", "patterns": [r"^empty"]},
        {"label": "Broken", "patterns": []},
        {"label": "Invalid", "patterns": [123]},
        {"patterns": [r"^missing label"]},
    ]

    parsed = _load_required_groups(json.dumps(custom))
    assert parsed == [{"label": "Lint", "patterns": [r"^lint /"]}]


def test_build_summary_comment_prefers_present_runs_when_duplicates() -> None:
    runs = [
        {"key": "ci", "displayName": "CI", "present": False, "jobs": []},
        {
            "key": "ci",
            "displayName": "CI",
            "present": True,
            "id": 55,
            "conclusion": "success",
            "html_url": "https://example.test/ci/55",
            "jobs": [],
        },
        {"key": "docker", "displayName": "Docker", "present": False, "jobs": []},
        {
            "key": "docker",
            "displayName": "Docker",
            "present": True,
            "id": 91,
            "status": "in_progress",
            "html_url": "https://example.test/docker/91",
            "jobs": [],
        },
    ]

    body = build_summary_comment(
        runs=runs,
        head_sha=None,
        coverage_stats=None,
        coverage_section=None,
        required_groups_env=None,
    )

    latest_line = next(
        (line for line in body.splitlines() if line.startswith("**Latest Runs:**")),
        "",
    )

    assert "CI (#55)" in latest_line
    assert "pending — CI" not in latest_line
    assert "⏳ in progress — [Docker (#91)]" in latest_line


def test_build_summary_comment_prefers_worse_state_for_duplicates() -> None:
    runs = [
        {
            "key": "ci",
            "displayName": "CI",
            "present": True,
            "id": 77,
            "conclusion": "success",
            "html_url": "https://example.test/ci/77",
            "jobs": [
                {
                    "name": "ci / python",
                    "conclusion": "success",
                    "html_url": "https://example.test/ci/77/python",
                }
            ],
        },
        {
            "key": "ci",
            "displayName": "CI",
            "present": True,
            "id": 78,
            "conclusion": "failure",
            "html_url": "https://example.test/ci/78",
            "jobs": [
                {
                    "name": "ci / python",
                    "conclusion": "failure",
                    "html_url": "https://example.test/ci/78/python",
                }
            ],
        },
    ]

    body = build_summary_comment(
        runs=runs,
        head_sha=None,
        coverage_stats=None,
        coverage_section=None,
        required_groups_env=None,
    )

    latest_line = next(
        (line for line in body.splitlines() if line.startswith("**Latest Runs:**")),
        "",
    )

    assert "❌ failure" in latest_line
    assert "(#78)" in latest_line
    assert "✅ success" not in latest_line
    assert "| **CI / ci / python** | ❌ failure | [logs]" in body
    assert "CI python: ❌ failure" in body


def test_main_appends_to_github_output(tmp_path, monkeypatch) -> None:
    output_file = tmp_path / "outputs.txt"
    output_file.write_text("existing=1\n", encoding="utf-8")

    monkeypatch.setenv("RUNS_JSON", "[]")
    monkeypatch.delenv("HEAD_SHA", raising=False)
    monkeypatch.delenv("COVERAGE_STATS", raising=False)
    monkeypatch.delenv("COVERAGE_SECTION", raising=False)
    monkeypatch.delenv("REQUIRED_JOB_GROUPS_JSON", raising=False)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    post_ci_summary.main()

    text = output_file.read_text(encoding="utf-8")
    assert text.startswith("existing=1\n"), "original content should be preserved"
    assert "body<<EOF" in text
    assert text.strip().endswith("EOF"), "output block should terminate with EOF marker"
