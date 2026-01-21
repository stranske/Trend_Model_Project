from __future__ import annotations

from trend.cli import _resolve_explain_output_paths
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
