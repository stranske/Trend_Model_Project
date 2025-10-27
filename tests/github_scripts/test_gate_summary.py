from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add script directory to path before importing gate_summary
SCRIPT_DIR = Path(__file__).resolve().parents[2] / ".github" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gate_summary  # noqa: E402


def write_summary(
    root: Path,
    runtime: str,
    *,
    format_outcome: str = "success",
    lint: str = "success",
    tests: str = "success",
    type_check: str = "success",
) -> None:
    summary_dir = root / "downloads" / runtime
    summary_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "python_version": runtime,
        "checks": {
            "format": {"outcome": format_outcome},
            "lint": {"outcome": lint},
            "tests": {"outcome": tests},
            "type_check": {"outcome": type_check},
            "coverage_minimum": {"outcome": "success"},
        },
        "coverage": {"percent": 91.23},
    }
    (summary_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_doc_only_summary_state() -> None:
    context = gate_summary.SummaryContext(
        doc_only=True,
        run_core=False,
        reason="docs_only",
        python_result="success",
        docker_result="skipped",
        docker_changed=False,
        artifacts_root=Path("/tmp/nonexistent"),
        summary_path=None,
        output_path=None,
    )

    result = gate_summary.summarize(context)
    assert result.state == "success"
    assert "docs-only" in "\n".join(result.lines)
    assert any("| docs-only | success |" in line for line in result.lines)
    assert result.format_failure is False


def test_active_summary_reads_artifacts(tmp_path: Path) -> None:
    write_summary(tmp_path, "3.11")

    context = gate_summary.SummaryContext(
        doc_only=False,
        run_core=True,
        reason="",
        python_result="success",
        docker_result="success",
        docker_changed=False,
        artifacts_root=tmp_path,
        summary_path=None,
        output_path=None,
    )

    result = gate_summary.summarize(context)
    joined = "\n".join(result.lines)
    assert result.state == "success"
    assert "Gate status" in joined
    assert "Reported coverage" in joined
    assert "Docker smoke skipped" in joined
    assert "| docker-smoke | success |" in joined
    assert result.cosmetic_failure is False
    assert result.format_failure is False


@pytest.mark.parametrize(
    "python_outcome, expected_state",
    [("failure", "failure"), ("success", "success")],
)
def test_summary_state_reflects_python_outcome(
    tmp_path: Path, python_outcome: str, expected_state: str
) -> None:
    write_summary(tmp_path, "3.12", lint="failure", tests=python_outcome)
    context = gate_summary.SummaryContext(
        doc_only=False,
        run_core=True,
        reason="",
        python_result=python_outcome,
        docker_result="success",
        docker_changed=True,
        artifacts_root=tmp_path,
        summary_path=None,
        output_path=None,
    )

    result = gate_summary.summarize(context)
    assert result.state == expected_state
    assert result.cosmetic_failure is False
    assert result.format_failure is False


def test_cosmetic_failure_detected(tmp_path: Path) -> None:
    write_summary(tmp_path, "3.11", format_outcome="failure")
    context = gate_summary.SummaryContext(
        doc_only=False,
        run_core=True,
        reason="",
        python_result="failure",
        docker_result="success",
        docker_changed=False,
        artifacts_root=tmp_path,
        summary_path=None,
        output_path=None,
    )

    result = gate_summary.summarize(context)
    assert result.state == "failure"
    assert result.cosmetic_failure is True
    assert result.failure_checks == ("format",)
    assert result.format_failure is True


def test_cosmetic_failure_rejects_other_failures(tmp_path: Path) -> None:
    write_summary(tmp_path, "3.11", tests="failure")
    context = gate_summary.SummaryContext(
        doc_only=False,
        run_core=True,
        reason="",
        python_result="failure",
        docker_result="success",
        docker_changed=False,
        artifacts_root=tmp_path,
        summary_path=None,
        output_path=None,
    )

    result = gate_summary.summarize(context)
    assert result.state == "failure"
    assert result.cosmetic_failure is False
    assert result.format_failure is False


def test_cosmetic_failure_reports_all_allowed_checks(tmp_path: Path) -> None:
    write_summary(tmp_path, "3.11", format_outcome="failure")
    write_summary(tmp_path, "3.12", lint="failure")

    context = gate_summary.SummaryContext(
        doc_only=False,
        run_core=True,
        reason="",
        python_result="failure",
        docker_result="success",
        docker_changed=False,
        artifacts_root=tmp_path,
        summary_path=None,
        output_path=None,
    )

    result = gate_summary.summarize(context)
    assert result.state == "failure"
    assert result.cosmetic_failure is True
    assert result.failure_checks == ("format", "lint")
    assert result.format_failure is True
