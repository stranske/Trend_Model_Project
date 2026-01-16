from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path

import pytest

from tools import eval_config_patch, prompt_evaluator
from tools.eval_config_patch import (
    DEFAULT_CASES,
    EvalResult,
    _evaluate_case,
    _format_summary_table,
    _load_cases,
)


def _find_case(case_id: str) -> dict[str, object]:
    for case in DEFAULT_CASES:
        if case.get("id") == case_id:
            return case
    raise AssertionError(f"Missing case: {case_id}")


def test_default_cases_include_edge_and_error_tags() -> None:
    assert len(DEFAULT_CASES) >= 10
    tag_counts = {"edge": 0, "error": 0}
    for case in DEFAULT_CASES:
        for tag in case.get("case_tags", []):
            if tag in tag_counts:
                tag_counts[tag] += 1
    assert tag_counts["edge"] >= 2
    assert tag_counts["error"] >= 2


def test_evaluate_case_passes_with_code_fence_response() -> None:
    case = _find_case("code_fenced_response")
    result = _evaluate_case(case)
    assert result.passed


def test_evaluate_case_missing_instruction_errors() -> None:
    result = _evaluate_case(
        {
            "id": "missing_instruction",
            "current_config": {},
            "expected_patch": {"operations": [], "risk_flags": [], "summary": "No changes."},
        }
    )
    assert not result.passed
    assert "Missing instruction." in result.errors


def test_evaluate_prompt_rejects_empty_llm_responses() -> None:
    case = {
        "id": "empty_llm_responses",
        "instruction": "Set top_n to 9.",
        "current_config": {"analysis": {"top_n": 8}},
        "expected_patch": {"operations": [], "risk_flags": [], "summary": "No changes."},
        "llm_responses": [],
    }

    result = eval_config_patch.evaluate_prompt(case, chain=None, mode="mock")

    assert not result.passed
    assert "llm_responses must be a non-empty list." in result.errors


def test_evaluate_prompt_rejects_non_string_llm_responses() -> None:
    case = {
        "id": "non_string_llm_responses",
        "instruction": "Set top_n to 9.",
        "current_config": {"analysis": {"top_n": 8}},
        "expected_patch": {"operations": [], "risk_flags": [], "summary": "No changes."},
        "llm_responses": [{"operations": []}],
    }

    result = eval_config_patch.evaluate_prompt(case, chain=None, mode="mock")

    assert not result.passed
    assert "llm_responses must contain only strings." in result.errors


def test_evaluate_prompt_rejects_non_string_llm_response() -> None:
    case = {
        "id": "non_string_llm_response",
        "instruction": "Set top_n to 9.",
        "current_config": {"analysis": {"top_n": 8}},
        "expected_patch": {"operations": [], "risk_flags": [], "summary": "No changes."},
        "llm_response": {"operations": []},
    }

    result = eval_config_patch.evaluate_prompt(case, chain=None, mode="mock")

    assert not result.passed
    assert "llm_response must be a string." in result.errors


def test_evaluate_prompt_rejects_conflicting_response_inputs() -> None:
    case = {
        "id": "conflicting_llm_response_inputs",
        "instruction": "Set top_n to 9.",
        "current_config": {"analysis": {"top_n": 8}},
        "expected_patch": {"operations": [], "risk_flags": [], "summary": "No changes."},
        "llm_response": json.dumps({"operations": []}, ensure_ascii=True),
        "llm_responses": [json.dumps({"operations": []}, ensure_ascii=True)],
    }

    result = eval_config_patch.evaluate_prompt(case, chain=None, mode="mock")

    assert not result.passed
    assert "Provide only one of llm_response or llm_responses." in result.errors


def test_evaluate_prompt_rejects_invalid_expected_log_fragments() -> None:
    case = {
        "id": "bad_expected_log_fragments",
        "instruction": "Set top_n to 9.",
        "current_config": {"analysis": {"top_n": 8}},
        "expected_patch": {"operations": [], "risk_flags": [], "summary": "No changes."},
        "llm_response": json.dumps({"operations": []}, ensure_ascii=True),
        "expected_log_fragments": "not-a-list",
    }

    result = eval_config_patch.evaluate_prompt(case, chain=None, mode="mock")

    assert not result.passed
    assert "expected_log_fragments must be a list." in result.errors


def test_evaluate_prompt_rejects_non_string_expected_log_fragments() -> None:
    case = {
        "id": "non_string_expected_log_fragments",
        "instruction": "Set top_n to 9.",
        "current_config": {"analysis": {"top_n": 8}},
        "expected_patch": {"operations": [], "risk_flags": [], "summary": "No changes."},
        "llm_response": json.dumps({"operations": []}, ensure_ascii=True),
        "expected_log_fragments": ["good", 123],
    }

    result = eval_config_patch.evaluate_prompt(case, chain=None, mode="mock")

    assert not result.passed
    assert "expected_log_fragments must contain only strings." in result.errors


def test_evaluate_prompt_rejects_non_integer_expected_log_count() -> None:
    case = {
        "id": "non_int_expected_log_count",
        "instruction": "Set top_n to 9.",
        "current_config": {"analysis": {"top_n": 8}},
        "expected_patch": {"operations": [], "risk_flags": [], "summary": "No changes."},
        "llm_response": json.dumps({"operations": []}, ensure_ascii=True),
        "expected_log_count": "one",
    }

    result = eval_config_patch.evaluate_prompt(case, chain=None, mode="mock")

    assert not result.passed
    assert "expected_log_count must be an integer." in result.errors


def test_evaluate_case_accepts_prompt_dataset_format() -> None:
    result = _evaluate_case(
        {
            "id": "prompt_format_case",
            "instruction": "Use risk parity weighting.",
            "starting_config": "config/defaults.yml",
            "expected_operations": [
                {
                    "op": "set",
                    "path": "portfolio.weighting_scheme",
                    "value": "risk_parity",
                }
            ],
        }
    )
    assert result.passed


def test_evaluate_prompt_mock_mode_runs_under_ten_seconds() -> None:
    case = _find_case("risk_parity_weighting")
    start = time.perf_counter()
    result = eval_config_patch.evaluate_prompt(case, chain=None, mode="mock")
    elapsed = time.perf_counter() - start
    assert result.passed
    assert elapsed < 10.0


def test_evaluate_prompt_mock_mode_benchmark_under_ten_seconds() -> None:
    case = _find_case("risk_parity_weighting")
    timings: list[float] = []
    for _ in range(3):
        start = time.perf_counter()
        result = eval_config_patch.evaluate_prompt(case, chain=None, mode="mock")
        timings.append(time.perf_counter() - start)
        assert result.passed
    assert max(timings) < 10.0


def test_evaluate_prompt_mock_mode_timeout_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    case = _find_case("risk_parity_weighting")
    ticks = [0.0, 11.0]

    def _fake_perf_counter() -> float:
        if ticks:
            return ticks.pop(0)
        return 11.0

    monkeypatch.setattr(prompt_evaluator.time, "perf_counter", _fake_perf_counter)
    result = eval_config_patch.evaluate_prompt(case, chain=None, mode="mock")
    assert not result.passed
    assert any("Mock mode execution exceeded 10 seconds" in error for error in result.errors)


def test_cli_live_mode_flag_switches_to_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def _fake_evaluate_prompt(case, chain, mode):
        calls.append(mode)
        return eval_config_patch.EvalResult(case_id=str(case["id"]), passed=True, errors=[])

    monkeypatch.setattr(eval_config_patch, "evaluate_prompt", _fake_evaluate_prompt)
    monkeypatch.setattr(eval_config_patch, "_build_live_chain", lambda **_kwargs: object())
    report_path = tmp_path / "report.json"

    exit_code = eval_config_patch.main(
        ["--use-default-cases", "--live-mode", "--report", str(report_path)]
    )

    assert exit_code == 0
    assert calls
    assert all(mode == "live" for mode in calls)


def test_evaluate_prompt_expected_error_must_fail_when_successful() -> None:
    case = {
        "id": "expected_error_not_raised",
        "instruction": "Set top_n to 9.",
        "current_config": {"analysis": {"top_n": 8}},
        "allowed_schema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "object", "properties": {"top_n": {"type": "integer"}}}
            },
        },
        "expected_patch": {
            "operations": [{"op": "set", "path": "analysis.top_n", "value": 9}],
            "risk_flags": [],
            "summary": "Set top_n to 9.",
        },
        "expected_error_contains": "Failed to parse ConfigPatch",
        "llm_response": json.dumps(
            {
                "operations": [{"op": "set", "path": "analysis.top_n", "value": 9}],
                "risk_flags": [],
                "summary": "Set top_n to 9.",
            },
            ensure_ascii=True,
        ),
    }

    result = eval_config_patch.evaluate_prompt(case, chain=None, mode="mock")

    assert not result.passed
    assert any("Expected error containing" in error for error in result.errors)


def test_evaluate_prompt_constraint_checks_pass() -> None:
    case = {
        "id": "constraints_pass",
        "instruction": "Set top_n to 9.",
        "current_config": {"analysis": {"top_n": 8}},
        "allowed_schema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "object", "properties": {"top_n": {"type": "integer"}}}
            },
        },
        "expected_patch": {
            "operations": [{"op": "set", "path": "analysis.top_n", "value": 9}],
            "risk_flags": [],
            "summary": "Set top_n to 9.",
        },
        "constraints": ["patch.operations | length == 1", "not patch.risk_flags"],
        "llm_response": json.dumps(
            {
                "operations": [{"op": "set", "path": "analysis.top_n", "value": 9}],
                "risk_flags": [],
                "summary": "Set top_n to 9.",
            },
            ensure_ascii=True,
        ),
    }

    result = eval_config_patch.evaluate_prompt(case, chain=None, mode="mock")

    assert result.passed


def test_evaluate_prompt_constraint_checks_fail() -> None:
    case = {
        "id": "constraints_fail",
        "instruction": "Set top_n to 9.",
        "current_config": {"analysis": {"top_n": 8}},
        "allowed_schema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "object", "properties": {"top_n": {"type": "integer"}}}
            },
        },
        "expected_patch": {
            "operations": [{"op": "set", "path": "analysis.top_n", "value": 9}],
            "risk_flags": [],
            "summary": "Set top_n to 9.",
        },
        "constraints": ["patch.operations | length == 2"],
        "llm_response": json.dumps(
            {
                "operations": [{"op": "set", "path": "analysis.top_n", "value": 9}],
                "risk_flags": [],
                "summary": "Set top_n to 9.",
            },
            ensure_ascii=True,
        ),
    }

    result = eval_config_patch.evaluate_prompt(case, chain=None, mode="mock")

    assert not result.passed
    assert any(
        "Constraint failed: patch.operations | length == 2" in error for error in result.errors
    )


def test_format_summary_table_includes_failure_diagnostics() -> None:
    results = [
        EvalResult(case_id="case_pass", passed=True, errors=[], duration=0.12),
        EvalResult(
            case_id="case_fail",
            passed=False,
            errors=["Bad output"],
            logs=["ConfigPatch parse attempt 1/2 failed"],
            duration=0.34,
        ),
    ]
    table = _format_summary_table(results)
    assert "case_pass" in table
    assert "PASS" in table
    assert "case_fail" in table
    assert "FAIL" in table
    assert "Time(s)" in table
    assert "Errors" in table
    assert "Warnings" in table
    assert "1: Bad output" in table
    assert "1: ConfigPatch parse attempt 1/2 failed" in table


def test_load_cases_clones_shared_yaml_anchors(tmp_path: Path) -> None:
    cases_file = tmp_path / "cases.yml"
    cases_file.write_text(
        textwrap.dedent(
            """
            base_config: &base_config
              analysis:
                top_n: 8
            base_schema: &base_schema
              type: object
              properties:
                analysis:
                  type: object
                  properties:
                    top_n:
                      type: integer
            cases:
              - id: case_one
                instruction: "Set top_n to 9."
                current_config: *base_config
                allowed_schema: *base_schema
                expected_patch:
                  operations: []
                  risk_flags: []
                  summary: "No changes."
              - id: case_two
                instruction: "Set top_n to 10."
                current_config: *base_config
                allowed_schema: *base_schema
                expected_patch:
                  operations: []
                  risk_flags: []
                  summary: "No changes."
            """
        ).lstrip(),
        encoding="utf-8",
    )

    cases = _load_cases(cases_file)
    cases[0]["current_config"]["analysis"]["top_n"] = 12
    cases[0]["allowed_schema"]["properties"]["analysis"]["properties"]["top_n"]["type"] = "number"

    assert cases[1]["current_config"]["analysis"]["top_n"] == 8
    assert (
        cases[1]["allowed_schema"]["properties"]["analysis"]["properties"]["top_n"]["type"]
        == "integer"
    )


def test_cli_min_cases_enforced(tmp_path: Path) -> None:
    cases_file = tmp_path / "cases.yml"
    cases_file.write_text(
        textwrap.dedent(
            """
            cases:
              - id: single_case
                instruction: "Set top_n to 9."
                current_config:
                  analysis:
                    top_n: 8
                expected_patch:
                  operations: []
                  risk_flags: []
                  summary: "No changes."
            """
        ).lstrip(),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    exit_code = eval_config_patch.main(
        ["--cases", str(cases_file), "--min-cases", "2", "--report", str(report_path)]
    )

    assert exit_code == 1
    assert not report_path.exists()
