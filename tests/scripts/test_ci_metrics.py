from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from scripts import ci_metrics as cm


class _FrozenDatetime(datetime):
    """datetime subclass with deterministic ``utcnow``."""

    @classmethod
    def utcnow(cls) -> datetime:  # type: ignore[override]
        return datetime(2024, 8, 22, 17, 45, 0)


@pytest.fixture
def junit_report(tmp_path: Path) -> Path:
    root = ET.Element("testsuite")

    # Passing case with namespace-less tags
    ET.SubElement(root, "testcase", name="test_ok", classname="suite.case", time="0.35")

    # Failure case with namespaced tag and text payload
    failing = ET.SubElement(
        root, "testcase", name="test_fail", classname="suite.case", time="1.25"
    )
    failure = ET.SubElement(
        failing, "{http://pytest}failure", message="boom", type="AssertionError"
    )
    failure.text = "  detailed explanation  "

    # Error case with malformed time to exercise fallback handling
    errored = ET.SubElement(
        root, "testcase", name="test_error", classname="suite.case", time="not-a-number"
    )
    ET.SubElement(errored, "error", message="err", type="RuntimeError").text = "stack"

    # Skipped case without classname to trigger fallback nodeid logic
    skipped = ET.SubElement(root, "testcase", name="", classname="", time="0.05")
    ET.SubElement(skipped, "skipped", message="not relevant")

    # Ensure sorted slow-test tie-breaking uses nodeid ordering
    slow_tie_a = ET.SubElement(
        root, "testcase", name="test_slow_a", classname="mod.A", time="2.0"
    )
    slow_tie_b = ET.SubElement(
        root, "testcase", name="test_slow_b", classname="mod.B", time="2.0"
    )
    ET.SubElement(slow_tie_b, "{http://pytest}skipped")

    path = tmp_path / "report.xml"
    ET.ElementTree(root).write(path, encoding="utf-8")
    return path


def test_tag_name_strips_namespace() -> None:
    namespaced = ET.Element("{http://example.com}failure")
    assert cm._tag_name(namespaced) == "failure"
    plain = ET.Element("skipped")
    assert cm._tag_name(plain) == "skipped"


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [(None, 7, 7), ("", 4, 4), ("12", 0, 12)],
)
def test_parse_int_accepts_defaults_and_values(
    value: str | None, default: int, expected: int
) -> None:
    assert cm._parse_int(value, "TOP_N", default) == expected


@pytest.mark.parametrize("bad", ["-1", "-10"])
def test_parse_int_rejects_negative_values(bad: str) -> None:
    with pytest.raises(SystemExit):
        cm._parse_int(bad, "TOP_N", 0)


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [(None, 1.5, 1.5), ("", 0.5, 0.5), ("2.75", 1.0, 2.75)],
)
def test_parse_float_accepts_defaults_and_values(
    value: str | None, default: float, expected: float
) -> None:
    assert cm._parse_float(value, "MIN_SECONDS", default) == expected


@pytest.mark.parametrize("bad", ["-0.1", "-4.0"])
def test_parse_float_rejects_negative_values(bad: str) -> None:
    with pytest.raises(SystemExit):
        cm._parse_float(bad, "MIN_SECONDS", 0.0)


@pytest.mark.parametrize(
    ("classname", "name", "expected"),
    [
        ("pkg.module", "test_case", "pkg.module::test_case"),
        ("pkg.module", "", "pkg.module"),
        ("", "test_case", "test_case"),
        ("", "", "(unknown)"),
    ],
)
def test_build_nodeid(classname: str, name: str, expected: str) -> None:
    assert cm._build_nodeid(classname, name) == expected


def test_extract_testcases_and_summaries(junit_report: Path) -> None:
    root = ET.parse(junit_report).getroot()
    cases = cm._extract_testcases(root)
    assert {case.outcome for case in cases} == {"passed", "failure", "error", "skipped"}
    malformed_time = next(c for c in cases if c.name == "test_error")
    assert malformed_time.time == 0.0  # ValueError fallback

    summary = cm._summarise(cases)
    assert summary == {
        "tests": len(cases),
        "failures": 1,
        "errors": 1,
        "skipped": 2,
        "passed": len(cases) - 4,
        "duration_seconds": pytest.approx(sum(c.time for c in cases)),
    }

    failure_rows = cm._collect_failures(cases)
    assert failure_rows == [
        {
            "status": "failure",
            "name": "test_fail",
            "classname": "suite.case",
            "nodeid": "suite.case::test_fail",
            "time": 1.25,
            "message": "boom",
            "type": "AssertionError",
            "details": "detailed explanation",
        },
        {
            "status": "error",
            "name": "test_error",
            "classname": "suite.case",
            "nodeid": "suite.case::test_error",
            "time": 0.0,
            "message": "err",
            "type": "RuntimeError",
            "details": "stack",
        },
    ]


def test_collect_slow_tests_handles_threshold_and_sorting(junit_report: Path) -> None:
    cases = cm._extract_testcases(ET.parse(junit_report).getroot())
    slow_tests = cm._collect_slow_tests(cases, top_n=3, min_seconds=0.5)
    assert [item["nodeid"] for item in slow_tests] == [
        "mod.A::test_slow_a",
        "mod.B::test_slow_b",
        "suite.case::test_fail",
    ]

    assert cm._collect_slow_tests(cases, top_n=0, min_seconds=0.0) == []


def test_build_metrics_produces_expected_payload(
    junit_report: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cm._dt, "datetime", _FrozenDatetime)
    payload = cm.build_metrics(junit_report, top_n=3, min_seconds=0.5)
    assert payload["generated_at"] == "2024-08-22T17:45:00Z"
    assert payload["junit_path"].endswith("report.xml")
    assert payload["summary"]["tests"] == 6
    assert payload["failures"][0]["nodeid"] == "suite.case::test_fail"
    assert payload["slow_tests"] == {
        "threshold_seconds": 0.5,
        "limit": 3,
        "items": [
            {
                "name": "test_slow_a",
                "classname": "mod.A",
                "nodeid": "mod.A::test_slow_a",
                "time": 2.0,
                "outcome": "passed",
            },
            {
                "name": "test_slow_b",
                "classname": "mod.B",
                "nodeid": "mod.B::test_slow_b",
                "time": 2.0,
                "outcome": "skipped",
            },
            {
                "name": "test_fail",
                "classname": "suite.case",
                "nodeid": "suite.case::test_fail",
                "time": 1.25,
                "outcome": "failure",
            },
        ],
    }


def test_build_metrics_requires_existing_report(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cm.build_metrics(tmp_path / "missing.xml")


def test_main_writes_metrics_file(
    junit_report: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cm._dt, "datetime", _FrozenDatetime)
    output_path = tmp_path / "metrics.json"
    monkeypatch.setenv("JUNIT_PATH", str(junit_report))
    monkeypatch.setenv("OUTPUT_PATH", str(output_path))
    monkeypatch.setenv("TOP_N", "2")
    monkeypatch.setenv("MIN_SECONDS", "0.25")

    try:
        exit_code = cm.main()
    finally:
        monkeypatch.delenv("JUNIT_PATH", raising=False)
        monkeypatch.delenv("OUTPUT_PATH", raising=False)
        monkeypatch.delenv("TOP_N", raising=False)
        monkeypatch.delenv("MIN_SECONDS", raising=False)

    assert exit_code == 0
    data = json.loads(output_path.read_text())
    assert data["summary"]["tests"] == 6
    captured = capsys.readouterr()
    assert f"Metrics written to {output_path}" in captured.out


def test_main_reports_missing_junit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.xml"
    output = tmp_path / "unused.json"
    monkeypatch.setenv("JUNIT_PATH", str(missing))
    monkeypatch.setenv("OUTPUT_PATH", str(output))
    try:
        exit_code = cm.main()
    finally:
        monkeypatch.delenv("JUNIT_PATH", raising=False)
        monkeypatch.delenv("OUTPUT_PATH", raising=False)

    assert exit_code == 1
    assert not output.exists()
    captured = capsys.readouterr()
    assert "JUnit report not found" in captured.err
