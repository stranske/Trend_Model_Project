import json
from pathlib import Path

import pytest

from scripts import ci_history


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, False),
        ("", False),
        ("0", False),
        ("Yes", True),
        ("TRUE", True),
    ],
)
def test_truthy(value: str | None, expected: bool) -> None:
    assert ci_history._truthy(value) is expected


def test_load_metrics_prefers_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    junit_path = tmp_path / "report.xml"
    junit_path.write_text("<testsuite/>", encoding="utf-8")
    metrics_path = tmp_path / "metrics.json"
    metrics_data = {"summary": {"passed": 10}, "slow_tests": ["a"]}
    metrics_path.write_text(json.dumps(metrics_data), encoding="utf-8")

    monkeypatch.setattr(ci_history.ci_metrics, "build_metrics", lambda _: {"summary": {}})

    data, from_file = ci_history._load_metrics(junit_path, metrics_path)

    assert data == metrics_data
    assert from_file is True


def test_load_metrics_regenerates_when_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    junit_path = tmp_path / "junit.xml"
    junit_path.write_text("<testsuite/>", encoding="utf-8")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text("{", encoding="utf-8")

    generated = {"summary": {"failed": 1}}
    monkeypatch.setattr(ci_history.ci_metrics, "build_metrics", lambda path: generated)

    data, from_file = ci_history._load_metrics(junit_path, metrics_path)

    assert data == generated
    assert from_file is False


def test_load_metrics_when_file_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    junit_path = tmp_path / "report.xml"
    junit_path.write_text("<testsuite/>", encoding="utf-8")
    metrics_path = tmp_path / "missing.json"

    generated = {"summary": {"passed": 2}}
    monkeypatch.setattr(ci_history.ci_metrics, "build_metrics", lambda path: generated)

    data, from_file = ci_history._load_metrics(junit_path, metrics_path)

    assert metrics_path.exists() is False
    assert data == generated
    assert from_file is False


def test_load_metrics_regenerates_when_summary_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    junit_path = tmp_path / "report.xml"
    junit_path.write_text("<testsuite/>", encoding="utf-8")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"other": {}}), encoding="utf-8")

    generated = {"summary": {"passed": 1}}
    monkeypatch.setattr(ci_history.ci_metrics, "build_metrics", lambda path: generated)

    data, from_file = ci_history._load_metrics(junit_path, metrics_path)

    assert data == generated
    assert from_file is False


def test_build_history_record_includes_optional_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = {
        "summary": {"passed": 5},
        "failures": [
            {"status": "failed", "nodeid": "tests/test_demo.py::test_fail"}
        ],
        "slow_tests": [
            {"nodeid": "tests/test_demo.py::test_slow", "time": 12.3}
        ],
    }
    junit_path = Path("/tmp/junit.xml")
    metrics_path = Path("/tmp/metrics.json")

    monkeypatch.setenv("GITHUB_RUN_ID", "101")
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "7")
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")

    record = ci_history._build_history_record(
        metrics,
        junit_path=junit_path,
        metrics_path=metrics_path,
        metrics_from_file=True,
    )

    assert record["summary"] == metrics["summary"]
    assert record["failures"] == metrics["failures"]
    assert record["slow_tests"] == metrics["slow_tests"]
    assert record["metrics_path"] == str(metrics_path)
    assert record["github"] == {
        "github_run_id": "101",
        "github_run_number": "7",
        "github_sha": "deadbeef",
        "github_ref": "refs/heads/main",
    }


def test_build_classification_payload_counts_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = {
        "failures": [
            {"status": "failed", "nodeid": "t::a", "message": "boom", "type": "AssertionError", "time": 1.0},
            {"status": "error", "nodeid": "t::b", "message": "zap", "type": "RuntimeError", "time": 2.0},
            {"status": "failed", "nodeid": "t::c", "message": "kapow", "type": "AssertionError", "time": 3.0},
        ]
    }

    payload = ci_history._build_classification_payload(metrics)

    assert payload["counts"] == {"failed": 2, "error": 1}
    assert payload["total"] == 3
    assert len(payload["entries"]) == 3


def test_main_appends_history_and_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    junit_path = tmp_path / "report.xml"
    junit_path.write_text("<testsuite/>", encoding="utf-8")
    metrics_path = tmp_path / "metrics.json"
    metrics_data = {
        "summary": {"passed": 12, "failed": 1},
        "failures": [
            {"status": "failed", "nodeid": "tests::test_fail", "message": "boom", "type": "AssertionError"}
        ],
        "slow_tests": [{"nodeid": "tests::test_slow", "time": 4.2}],
    }
    metrics_path.write_text(json.dumps(metrics_data), encoding="utf-8")
    history_path = tmp_path / "history.ndjson"
    classification_path = tmp_path / "classification.json"

    monkeypatch.setenv("JUNIT_PATH", str(junit_path))
    monkeypatch.setenv("METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("HISTORY_PATH", str(history_path))
    monkeypatch.setenv("CLASSIFICATION_OUT", str(classification_path))
    monkeypatch.setenv("ENABLE_CLASSIFICATION", "true")
    monkeypatch.setenv("GITHUB_RUN_ID", "202")
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "9")
    monkeypatch.setenv("GITHUB_SHA", "cafebabe")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/feature")

    exit_code = ci_history.main()

    assert exit_code == 0
    history_lines = history_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(history_lines) == 1
    record = json.loads(history_lines[0])
    assert record["summary"] == metrics_data["summary"]
    assert record["failures"] == metrics_data["failures"]
    assert record["slow_tests"] == metrics_data["slow_tests"]
    assert record["metrics_path"] == str(metrics_path)
    assert record["github"]["github_run_id"] == "202"

    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    assert classification["total"] == 1
    assert classification["counts"] == {"failed": 1}

    output = capsys.readouterr().out
    assert "History appended" in output
    assert "Classification written" in output


def test_main_handles_regenerated_metrics_and_cleans_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    junit_path = tmp_path / "report.xml"
    junit_path.write_text("<testsuite/>", encoding="utf-8")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text("{", encoding="utf-8")
    history_path = tmp_path / "history.ndjson"
    classification_path = tmp_path / "classification.json"
    classification_path.write_text("stale", encoding="utf-8")

    generated_metrics = {"summary": {"passed": 3}, "failures": []}
    monkeypatch.setattr(
        ci_history.ci_metrics,
        "build_metrics",
        lambda _: generated_metrics,
    )

    monkeypatch.delenv("ENABLE_CLASSIFICATION", raising=False)
    monkeypatch.setenv("ENABLE_CLASSIFICATION_FLAG", "off")
    monkeypatch.setenv("JUNIT_PATH", str(junit_path))
    monkeypatch.setenv("METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("HISTORY_PATH", str(history_path))
    monkeypatch.setenv("CLASSIFICATION_OUT", str(classification_path))

    exit_code = ci_history.main()

    assert exit_code == 0
    record = json.loads(history_path.read_text(encoding="utf-8").strip())
    assert "metrics_path" not in record
    assert classification_path.exists() is False


def test_main_skips_classification_cleanup_when_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    junit_path = tmp_path / "report.xml"
    junit_path.write_text("<testsuite/>", encoding="utf-8")
    metrics_path = tmp_path / "metrics.json"
    metrics_data = {"summary": {"passed": 2}, "failures": []}
    metrics_path.write_text(json.dumps(metrics_data), encoding="utf-8")
    history_path = tmp_path / "history.ndjson"
    classification_path = tmp_path / "classification.json"

    monkeypatch.setenv("JUNIT_PATH", str(junit_path))
    monkeypatch.setenv("METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("HISTORY_PATH", str(history_path))
    monkeypatch.setenv("CLASSIFICATION_OUT", str(classification_path))
    monkeypatch.setenv("ENABLE_CLASSIFICATION_FLAG", "0")
    monkeypatch.delenv("ENABLE_CLASSIFICATION", raising=False)

    exit_code = ci_history.main()

    assert exit_code == 0
    assert classification_path.exists() is False


def test_main_missing_junit_reports_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("JUNIT_PATH", "nonexistent.xml")
    monkeypatch.delenv("METRICS_PATH", raising=False)
    monkeypatch.delenv("HISTORY_PATH", raising=False)
    monkeypatch.delenv("CLASSIFICATION_OUT", raising=False)
    monkeypatch.delenv("ENABLE_CLASSIFICATION", raising=False)

    exit_code = ci_history.main()

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "JUnit report not found" in err


def test_main_reports_missing_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    junit_path = tmp_path / "report.xml"
    junit_path.write_text("<testsuite/>", encoding="utf-8")
    metrics_path = tmp_path / "metrics.json"

    monkeypatch.setenv("JUNIT_PATH", str(junit_path))
    monkeypatch.setenv("METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("HISTORY_PATH", str(tmp_path / "history.ndjson"))
    monkeypatch.setenv("CLASSIFICATION_OUT", str(tmp_path / "classification.json"))
    monkeypatch.delenv("ENABLE_CLASSIFICATION", raising=False)
    monkeypatch.delenv("ENABLE_CLASSIFICATION_FLAG", raising=False)

    def raise_missing(*args, **kwargs):
        raise FileNotFoundError("metrics missing")

    monkeypatch.setattr(ci_history, "_load_metrics", raise_missing)

    exit_code = ci_history.main()

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "metrics missing" in captured.err
