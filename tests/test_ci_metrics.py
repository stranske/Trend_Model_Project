from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import ci_metrics


def _write_sample_junit(tmp_path: Path) -> Path:
    junit_xml = """
    <testsuite name="pytest" tests="4" failures="1" errors="1" skipped="1" time="4.2">
      <testcase classname="pkg.test_mod" name="test_ok" time="0.501" />
      <testcase classname="pkg.test_mod" name="test_fail" time="1.5">
        <failure message="assert false" type="AssertionError">stacktrace here</failure>
      </testcase>
      <testcase classname="pkg.test_mod" name="test_err" time="0.25">
        <error message="boom" type="RuntimeError">traceback</error>
      </testcase>
      <testcase classname="pkg.test_mod" name="test_skip" time="0.0">
        <skipped message="not needed" />
      </testcase>
    </testsuite>
    """.strip()
    junit_path = tmp_path / "pytest-junit.xml"
    junit_path.write_text(junit_xml, encoding="utf-8")
    return junit_path


def test_build_metrics_extracts_counts_and_failures(tmp_path: Path) -> None:
    junit_path = _write_sample_junit(tmp_path)

    payload = ci_metrics.build_metrics(junit_path, top_n=5, min_seconds=0.2)

    summary = payload["summary"]
    assert summary == {
        "tests": 4,
        "failures": 1,
        "errors": 1,
        "skipped": 1,
        "passed": 1,
        "duration_seconds": pytest.approx(2.251, rel=1e-6),
    }

    failure_entries = payload["failures"]
    assert len(failure_entries) == 2
    assert failure_entries[0] == {
        "status": "failure",
        "name": "test_fail",
        "classname": "pkg.test_mod",
        "nodeid": "pkg.test_mod::test_fail",
        "time": pytest.approx(1.5),
        "message": "assert false",
        "type": "AssertionError",
        "details": "stacktrace here",
    }
    assert failure_entries[1]["status"] == "error"
    assert failure_entries[1]["details"] == "traceback"


def test_build_metrics_slow_tests_filter(tmp_path: Path) -> None:
    junit_path = _write_sample_junit(tmp_path)

    payload = ci_metrics.build_metrics(junit_path, top_n=2, min_seconds=1.0)

    slow = payload["slow_tests"]
    assert slow["threshold_seconds"] == 1.0
    assert slow["limit"] == 2
    assert slow["items"] == [
        {
            "name": "test_fail",
            "classname": "pkg.test_mod",
            "nodeid": "pkg.test_mod::test_fail",
            "time": pytest.approx(1.5),
            "outcome": "failure",
        }
    ]


def test_main_writes_output_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    junit_path = _write_sample_junit(tmp_path)
    out_path = tmp_path / "ci-metrics.json"

    monkeypatch.setenv("JUNIT_PATH", str(junit_path))
    monkeypatch.setenv("OUTPUT_PATH", str(out_path))
    monkeypatch.setenv("TOP_N", "3")
    monkeypatch.setenv("MIN_SECONDS", "0")

    try:
        exit_code = ci_metrics.main()
    finally:
        monkeypatch.delenv("JUNIT_PATH", raising=False)
        monkeypatch.delenv("OUTPUT_PATH", raising=False)
        monkeypatch.delenv("TOP_N", raising=False)
        monkeypatch.delenv("MIN_SECONDS", raising=False)

    assert exit_code == 0
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["summary"]["tests"] == 4
    assert data["slow_tests"]["limit"] == 3


def test_main_missing_junit_returns_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    junit_path = tmp_path / "missing.xml"
    out_path = tmp_path / "unused.json"

    monkeypatch.setenv("JUNIT_PATH", str(junit_path))
    monkeypatch.setenv("OUTPUT_PATH", str(out_path))

    try:
        exit_code = ci_metrics.main()
    finally:
        monkeypatch.delenv("JUNIT_PATH", raising=False)
        monkeypatch.delenv("OUTPUT_PATH", raising=False)

    assert exit_code == 1
    assert not out_path.exists()


def test_invalid_top_n_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    junit_path = _write_sample_junit(tmp_path)
    out_path = tmp_path / "ci-metrics.json"

    monkeypatch.setenv("JUNIT_PATH", str(junit_path))
    monkeypatch.setenv("OUTPUT_PATH", str(out_path))
    monkeypatch.setenv("TOP_N", "-4")

    with pytest.raises(SystemExit):
        ci_metrics.main()

    monkeypatch.delenv("JUNIT_PATH", raising=False)
    monkeypatch.delenv("OUTPUT_PATH", raising=False)
    monkeypatch.delenv("TOP_N", raising=False)
