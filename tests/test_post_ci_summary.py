"""Unit tests for the post-CI summary helpers."""
from pathlib import Path

from tools.post_ci_summary import (
    CoverageDetails,
    FailureSnapshot,
    JobRecord,
    Requirement,
    WorkflowConfig,
    WorkflowSummary,
    build_comment_body,
    combine_states,
    load_coverage_details,
    load_failure_snapshot,
    render_coverage_section,
    render_job_table,
    summarize_requirements,
)


def _make_summary(name: str, jobs: list[JobRecord]) -> WorkflowSummary:
    config = WorkflowConfig(name=name, workflow_file=f"{name}.yml", requirements=(Requirement(f"{name} req", ("*",)),))
    return WorkflowSummary(config=config, run_id=1, html_url=f"https://example.com/{name}", conclusion="success", status="completed", jobs=jobs)


def test_combine_states_prioritises_failure() -> None:
    assert combine_states(["success", "failure", "success"]) == "failure"
    assert combine_states(["success", "skipped"]) == "success"
    assert combine_states(["skipped", "skipped"]) == "skipped"


def test_summarize_requirements_uses_badges() -> None:
    jobs = [JobRecord(name="job-1", conclusion="success", status="completed", html_url="https://example.com/job1")]
    summary = _make_summary("ci", jobs)
    lines = summarize_requirements(summary)
    assert lines == ["ci req: ✅ success"]


def test_render_job_table_orders_failed_rows_first() -> None:
    failing = WorkflowSummary(
        config=WorkflowConfig(name="CI", workflow_file="ci.yml", requirements=()),
        run_id=10,
        html_url="https://example.com/ci",
        conclusion="completed",
        status="completed",
        jobs=[JobRecord(name="tests", conclusion="failure", status="completed", html_url="https://example.com/ci/tests")],
    )
    passing = WorkflowSummary(
        config=WorkflowConfig(name="Docker", workflow_file="docker.yml", requirements=()),
        run_id=11,
        html_url="https://example.com/docker",
        conclusion="completed",
        status="completed",
        jobs=[JobRecord(name="smoke", conclusion="success", status="completed", html_url="https://example.com/docker/smoke")],
    )
    table = render_job_table([passing, failing])
    lines = table.splitlines()
    assert lines[2].startswith("| **CI — tests** | ❌ failure |")
    assert lines[3].startswith("| Docker — smoke | ✅ success |")


def test_render_coverage_section_includes_delta() -> None:
    details = CoverageDetails(avg_latest=48.123, avg_delta=0.5, worst_latest=12.0, worst_delta=-0.25, table_markdown="| h | v |\n| - | - |")
    section = render_coverage_section(details)
    assert section is not None
    assert "48.12%" in section
    assert "+0.50pp" in section
    assert "-0.25pp" in section
    assert "| h | v |" in section


def test_build_comment_body_combines_sections() -> None:
    ci_summary = WorkflowSummary(
        config=WorkflowConfig(name="CI", workflow_file="ci.yml", requirements=(Requirement("CI req", ("tests",)),)),
        run_id=1,
        html_url="https://example.com/ci",
        conclusion="completed",
        status="completed",
        jobs=[JobRecord(name="tests", conclusion="success", status="completed", html_url="https://example.com/ci/tests")],
    )
    docker_summary = WorkflowSummary(
        config=WorkflowConfig(name="Docker", workflow_file="docker.yml", requirements=(Requirement("Docker req", ("smoke",)),)),
        run_id=2,
        html_url="https://example.com/docker",
        conclusion="completed",
        status="completed",
        jobs=[JobRecord(name="smoke", conclusion="success", status="completed", html_url="https://example.com/docker/smoke")],
    )
    coverage = CoverageDetails(avg_latest=50.0, avg_delta=1.0, worst_latest=40.0, worst_delta=0.0, table_markdown="| h | v |\n| - | - |")
    snapshot = FailureSnapshot(issues=[{"number": 123, "occurrences": 2, "last_seen": "2024-01-01", "url": "https://example.com/issue/123"}])
    body = build_comment_body("deadbeef", [ci_summary, docker_summary], coverage, snapshot, "CI")
    assert "### Automated Status Summary" in body
    assert "**Required:**" in body
    assert "### Coverage (soft gate)" in body
    assert "### Open Failure Signatures" in body
    assert "deadbeef" in body


def test_load_coverage_details_reads_latest_and_deltas(tmp_path: Path) -> None:
    artifacts = tmp_path
    (artifacts / "coverage_summary.md").write_text("| file | % |\n| a | 50 |", encoding="utf-8")
    (artifacts / "coverage-trend.json").write_text(
        '{"avg_coverage": 48.5, "worst_job_coverage": 20.0}',
        encoding="utf-8",
    )
    (artifacts / "coverage-trend-history.json").write_text(
        (
            "[\n"
            "  {\"avg_coverage\": 47.0, \"worst_job_coverage\": 18.0},\n"
            "  {\"avg_coverage\": 48.5, \"worst_job_coverage\": 20.0}\n"
            "]"
        ),
        encoding="utf-8",
    )

    details = load_coverage_details(artifacts)

    assert details.table_markdown == "| file | % |\n| a | 50 |"
    assert details.avg_latest == 48.5
    assert details.worst_latest == 20.0
    assert details.avg_delta == 1.5
    assert details.worst_delta == 2.0


def test_load_failure_snapshot_filters_invalid_entries(tmp_path: Path) -> None:
    artifacts = tmp_path
    (artifacts / "ci_failures_snapshot.json").write_text(
        (
            "{\n"
            "  \"issues\": [\n"
            "    {\"number\": 101, \"url\": \"https://example.com\"},\n"
            "    \"not-a-dict\"\n"
            "  ]\n"
            "}\n"
        ),
        encoding="utf-8",
    )

    snapshot = load_failure_snapshot(artifacts)

    assert snapshot is not None
    assert snapshot.issues == [{"number": 101, "url": "https://example.com"}]
