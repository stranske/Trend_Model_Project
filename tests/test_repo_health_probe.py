from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import repo_health_probe


def test_run_probe_from_sources_success():
    report = repo_health_probe.run_probe_from_sources(
        labels=["agent:codex", "priority:p0", "tech:coverage", "workflows"],
        secrets=["SERVICE_BOT_PAT"],
        variables=["OPS_HEALTH_ISSUE"],
    )
    assert report["ok"] is True
    summary = repo_health_probe.build_summary(
        report, actionlint_ok=True, issue_id_present=True
    )
    assert summary.status == "success"
    assert "Workflow lint" not in "\n".join(summary.issue_lines)
    assert "OPS_HEALTH_ISSUE" not in summary.summary_markdown


def test_build_summary_failure_paths():
    report = repo_health_probe.run_probe_from_sources(
        labels=["agent:codex"],
        secrets=[],
        variables=[],
    )
    summary = repo_health_probe.build_summary(
        report, actionlint_ok=False, issue_id_present=False
    )
    assert summary.status == "failure"
    joined = "\n".join(summary.issue_lines)
    assert "Workflow lint (`actionlint`) failed" in joined
    assert any("SERVICE_BOT_PAT" in line for line in summary.issue_lines)
    assert "OPS_HEALTH_ISSUE" in summary.summary_markdown


@pytest.mark.parametrize(
    "fixture_payload, expected_ok",
    [
        (
            {
                "labels": ["agent:codex", "priority:p1", "workflows", "tech:coverage"],
                "secrets": ["SERVICE_BOT_PAT"],
                "variables": ["OPS_HEALTH_ISSUE"],
            },
            True,
        ),
        (
            {
                "labels": ["agent:codex"],
                "secrets": [],
                "variables": [],
            },
            False,
        ),
    ],
)
def test_main_with_fixtures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixture_payload, expected_ok):
    fixture_file = tmp_path / "fixture.json"
    fixture_file.write_text(json.dumps(fixture_payload), encoding="utf-8")
    report_path = tmp_path / "report.json"
    step_summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))

    exit_code = repo_health_probe.main(
        [str(report_path), "--fixtures", str(fixture_file), "--write-summary"]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is expected_ok
    assert exit_code == (0 if expected_ok else 1)
    summary_text = step_summary.read_text(encoding="utf-8")
    assert "Repo health nightly checks" in summary_text
