"""Unit tests for the post-CI summary helpers."""
from tools.post_ci_summary import (
    CoverageDetails,
    FailureSnapshot,
    JobRecord,
    Requirement,
    WorkflowConfig,
    WorkflowSummary,
    build_comment_body,
    combine_states,
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
