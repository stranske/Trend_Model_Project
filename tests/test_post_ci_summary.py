from __future__ import annotations

import json

import pytest

from tools.post_ci_summary import build_summary_comment


@pytest.fixture()
def sample_runs() -> list[dict[str, object]]:
    return [
        {
            "key": "gate",
            "displayName": "Gate",
            "present": True,
            "id": 101,
            "run_attempt": 1,
            "conclusion": "failure",
            "status": "completed",
            "html_url": "https://example.test/gate/101",
            "jobs": [
                {
                    "name": "Core Tests • py311",
                    "conclusion": "success",
                    "html_url": "https://example.test/gate/101/py311",
                },
                {
                    "name": "Core Tests • py312",
                    "conclusion": "success",
                    "html_url": "https://example.test/gate/101/py312",
                },
                {
                    "name": "Docker Smoke Check",
                    "conclusion": "failure",
                    "html_url": "https://example.test/gate/101/docker",
                },
                {
                    "name": "Maint Gate Aggregator",
                    "conclusion": "failure",
                    "html_url": "https://example.test/gate/101/gate",
                },
            ],
        }
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
                {"label": "Core Tests (3.11)", "patterns": [r"^Core Tests • py311$"]},
                {"label": "Core Tests (3.12)", "patterns": [r"^Core Tests • py312$"]},
                {"label": "Docker Smoke", "patterns": [r"^Docker Smoke Check$"]},
                {"label": "Gate Aggregator", "patterns": [r"^Maint Gate Aggregator$"]},
            ]
        ),
    )

    assert body.startswith("## Automated Status Summary")
    assert "**Head SHA:** abc123" in body
    assert (
        "**Latest Runs:** ❌ failure — [Gate (#101)](https://example.test/gate/101)"
        in body
    )
    assert "Core Tests • py311: ✅ success" in body
    assert "Core Tests • py312: ✅ success" in body
    assert "Docker Smoke Check: ❌ failure" in body
    assert "Maint Gate Aggregator: ❌ failure" in body
    assert "| Gate / Core Tests • py311 | ✅ success |" in body
    assert "| Gate / Core Tests • py312 | ✅ success |" in body
    assert "| **Gate / Docker Smoke Check** | ❌ failure |" in body
    assert "| **Gate / Maint Gate Aggregator** | ❌ failure |" in body
    # Coverage lines should render with percentages and deltas
    assert "Coverage (jobs): 91.23%" in body
    assert "Coverage (worst job): 83.11%" in body
    assert "Δ -0.77 pp" in body
    assert "Δ +1.02 pp" in body
    assert coverage_section in body


def test_build_summary_comment_handles_missing_runs_and_defaults() -> None:
    body = build_summary_comment(
        runs=[{"key": "gate", "displayName": "Gate", "present": False}],
        head_sha=None,
        coverage_stats=None,
        coverage_section=None,
        required_groups_env=None,
    )

    assert "Core Tests (3.11): ⏳ pending" in body
    assert "Core Tests (3.12): ⏳ pending" in body
    assert "Docker Smoke: ⏳ pending" in body
    assert "Gate Aggregator: ⏳ pending" in body
    assert "**Latest Runs:** ⏳ pending — Gate" in body
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
        (i for i, line in enumerate(table_lines) if "Docker Smoke Check" in line), None
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

    assert docker_index is not None, "'docker smoke' job not found in table_lines"
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


def test_build_summary_comment_handles_irregular_run_data() -> None:
    body = build_summary_comment(
        runs=[
            {
                "key": "gate",
                "displayName": "Gate",
                "present": True,
                "jobs": [
                    None,
                    {"name": "", "conclusion": None, "html_url": None},
                    {"name": "core tests (3.11)", "status": "queued"},
                ],
            },
            "not-a-mapping",
        ],  # type: ignore[arg-type]
        head_sha="def456",
        coverage_stats={},
        coverage_section=None,
        required_groups_env=json.dumps(
            [
                {
                    "label": "Core Tests (3.11)",
                    "patterns": [r"core\s*(tests?)?.*(3\.11|py\.?311)"],
                },
            ]
        ),
    )

    assert "**Head SHA:** def456" in body
    assert "**Latest Runs:** ⏳ pending — Gate" in body
    assert "core tests (3.11): ⏳ queued" in body
    assert "| Gate / core tests (3.11) | ⏳ queued | — |" in body
    assert "### Coverage Overview" not in body


def test_build_summary_comment_defaults_on_invalid_required_groups(
    sample_runs: list[dict[str, object]],
) -> None:
    body = build_summary_comment(
        runs=sample_runs,
        head_sha="deadbeef",
        coverage_stats=None,
        coverage_section=None,
        required_groups_env="{not-json}",
    )

    assert "Core Tests • py311: ✅ success" in body
    assert "Core Tests • py312: ✅ success" in body
    assert "Docker Smoke Check: ❌ failure" in body
    assert "Maint Gate Aggregator: ❌ failure" in body
