from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.coverage_trend import (
    Baseline,
    TrendResult,
    dump_artifact,
    evaluate_trend,
    load_baseline,
    read_coverage,
    write_github_output,
)


def write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_read_coverage_prefers_xml(tmp_path: Path) -> None:
    xml = tmp_path / "coverage.xml"
    write_file(
        xml,
        """<?xml version='1.0'?><coverage line-rate='0.875'></coverage>""",
    )
    json_path = tmp_path / "coverage.json"
    write_file(
        json_path, json.dumps({"totals": {"covered_lines": 30, "num_statements": 40}})
    )
    current = read_coverage(xml, json_path)
    assert current == pytest.approx(87.5, rel=1e-4)


def test_read_coverage_falls_back_to_json(tmp_path: Path) -> None:
    xml = tmp_path / "coverage.xml"
    json_path = tmp_path / "coverage.json"
    write_file(
        json_path, json.dumps({"totals": {"covered_lines": 45, "num_statements": 50}})
    )
    current = read_coverage(xml, json_path)
    assert current == pytest.approx(90.0, rel=1e-4)


def test_load_baseline_missing_file_uses_defaults(tmp_path: Path) -> None:
    baseline_path = tmp_path / "missing.json"
    baseline = load_baseline(baseline_path)
    assert baseline.line is None
    assert baseline.warn_drop == pytest.approx(1.0)


def test_evaluate_trend_warns_on_drop() -> None:
    baseline = Baseline(line=92.0, warn_drop=0.5)
    result = evaluate_trend(90.9, baseline)
    assert result.status == "warn"
    assert result.delta == pytest.approx(-1.1, rel=1e-3)
    comment = result.comment_body()
    assert "Coverage drop alert" in comment


def test_evaluate_trend_ok_when_within_tolerance() -> None:
    baseline = Baseline(line=92.0, warn_drop=2.0)
    result = evaluate_trend(91.0, baseline)
    assert result.status == "ok"
    assert result.delta == pytest.approx(-1.0, rel=1e-3)
    assert result.comment_body() == ""


def test_dump_artifact_and_outputs(tmp_path: Path) -> None:
    result = TrendResult(
        current=88.0,
        baseline=90.0,
        warn_drop=1.0,
        delta=-2.0,
        status="warn",
        minimum=85.0,
    )
    artifact_path = tmp_path / "artifact" / "coverage.json"
    dump_artifact(artifact_path, result)
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert data == {
        "current": 88.0,
        "baseline": 90.0,
        "delta": -2.0,
        "warn_drop": 1.0,
        "status": "warn",
    }
    output_path = tmp_path / "output.txt"
    write_github_output(output_path, result)
    output = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert "status=warn" in output
    assert any(line.startswith("comment<<EOF") for line in output)
