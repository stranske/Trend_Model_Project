from __future__ import annotations

import json
from datetime import datetime, timezone

from trend.cli import (
    _build_explain_artifact_payload,
    _finalize_explanation_text,
    _resolve_explain_output_paths,
    _write_explain_artifacts,
)
from trend_analysis.llm.result_validation import ResultClaimIssue, serialize_claim_issue


def test_serialize_claim_issue_json_safe() -> None:
    issue = ResultClaimIssue(
        kind="value_mismatch",
        message="Mismatch",
        detail={"ids": {1, 2}},
    )

    payload = serialize_claim_issue(issue)

    assert payload["kind"] == "value_mismatch"
    assert payload["message"] == "Mismatch"
    assert isinstance(payload["detail"]["ids"], str)


def test_resolve_explain_output_paths_directory(tmp_path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    txt_path, json_path = _resolve_explain_output_paths(out_dir, "abc123")

    assert txt_path == out_dir / "explanation_abc123.txt"
    assert json_path == out_dir / "explanation_abc123.json"


def test_resolve_explain_output_paths_prefix(tmp_path) -> None:
    prefix = tmp_path / "explanation_run.txt"

    txt_path, json_path = _resolve_explain_output_paths(prefix, "ignored")

    assert txt_path == tmp_path / "explanation_run.txt"
    assert json_path == tmp_path / "explanation_run.json"


def test_resolve_explain_output_paths_missing_directory(tmp_path) -> None:
    out_dir = tmp_path / "new_dir"

    txt_path, json_path = _resolve_explain_output_paths(out_dir, "run-42")

    assert txt_path == out_dir / "explanation_run-42.txt"
    assert json_path == out_dir / "explanation_run-42.json"


def test_build_explain_artifact_payload_serializes_claims() -> None:
    issue = ResultClaimIssue(
        kind="missing_citation",
        message="Missing citation",
        detail={"source": "out_sample_stats"},
    )

    payload = _build_explain_artifact_payload(
        run_id="run-123",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        text="Explanation text",
        metric_count=5,
        trace_url="trace://example",
        claim_issues=[issue],
    )

    assert payload["run_id"] == "run-123"
    assert payload["metric_count"] == 5
    assert payload["trace_url"] == "trace://example"
    assert payload["claim_issues"] == [serialize_claim_issue(issue)]


def test_write_explain_artifacts_creates_files(tmp_path) -> None:
    output = tmp_path / "artifacts"
    payload = _build_explain_artifact_payload(
        run_id="run-321",
        created_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
        text="Rendered explanation",
        metric_count=3,
        trace_url="trace://unit-test",
        claim_issues=[],
    )

    txt_path, json_path = _write_explain_artifacts(
        output=output,
        run_id="run-321",
        text="Rendered explanation",
        payload=payload,
    )

    assert txt_path.read_text(encoding="utf-8") == "Rendered explanation"
    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_payload["run_id"] == "run-321"
    assert json_payload["created_at"] == "2024-02-01T00:00:00+00:00"
    assert json_payload["text"] == "Rendered explanation"
    assert json_payload["metric_count"] == 3
    assert json_payload["trace_url"] == "trace://unit-test"
    assert json_payload["claim_issues"] == []


def test_finalize_explanation_text_adds_discrepancy_log_and_disclaimer() -> None:
    issue = ResultClaimIssue(
        kind="missing_citation",
        message="Missing citation",
        detail={"source": "out_sample_stats"},
    )

    output = _finalize_explanation_text("Base output", [issue])

    assert "Discrepancy log:" in output
    assert "This is analytical output, not financial advice." in output


def test_finalize_explanation_text_avoids_duplicate_log() -> None:
    issue = ResultClaimIssue(
        kind="missing_citation",
        message="Missing citation",
        detail={"source": "out_sample_stats"},
    )
    text = (
        "Base output\n\nDiscrepancy log:\n- missing_citation: Missing citation\n\n"
        "This is analytical output, not financial advice."
    )

    output = _finalize_explanation_text(text, [issue])

    assert output.count("Discrepancy log:") == 1
