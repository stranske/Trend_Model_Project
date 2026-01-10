from __future__ import annotations

from tools.eval_config_patch import DEFAULT_CASES, _evaluate_case


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
