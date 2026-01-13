from __future__ import annotations

import time

import pytest

from tools import eval_config_patch
from tools.eval_config_patch import (
    DEFAULT_CASES,
    EvalResult,
    _evaluate_case,
    _format_summary_table,
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


def test_evaluate_prompt_mock_mode_timeout_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    case = _find_case("risk_parity_weighting")
    ticks = [0.0, 11.0]

    def _fake_perf_counter() -> float:
        if ticks:
            return ticks.pop(0)
        return 11.0

    monkeypatch.setattr(eval_config_patch.time, "perf_counter", _fake_perf_counter)
    result = eval_config_patch.evaluate_prompt(case, chain=None, mode="mock")
    assert not result.passed
    assert any("Mock mode execution exceeded 10 seconds" in error for error in result.errors)


def test_format_summary_table_includes_failure_diagnostics() -> None:
    results = [
        EvalResult(case_id="case_pass", passed=True, errors=[]),
        EvalResult(
            case_id="case_fail",
            passed=False,
            errors=["Bad output"],
            logs=["ConfigPatch parse attempt 1/2 failed"],
        ),
    ]
    table = _format_summary_table(results)
    assert "case_pass" in table
    assert "PASS" in table
    assert "case_fail" in table
    assert "FAIL" in table
    assert "Bad output" in table
    assert "ConfigPatch parse attempt 1/2 failed" in table
