"""Integration tests for ConfigPatchChain eval cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("langchain_core")

from tools import prompt_evaluator  # noqa: E402

_EVAL_CASES_PATH = Path("tools/eval_test_cases.yml")


def _load_cases() -> list[dict[str, Any]]:
    payload = yaml.safe_load(_EVAL_CASES_PATH.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise TypeError("Eval cases payload must contain a list of cases.")
    return cases


_EVAL_CASES = _load_cases()


@pytest.mark.parametrize(
    "case",
    _EVAL_CASES,
    ids=[str(case.get("id", f"case-{index}")) for index, case in enumerate(_EVAL_CASES)],
)
def test_eval_cases_pass(case: dict[str, Any]) -> None:
    result = prompt_evaluator.evaluate_prompt(case, chain=None, mode="mock")
    assert result.passed, f"{case.get('id')} failed: {result.errors}"

    expected_patch = case.get("expected_patch") or {}
    expected_flags = expected_patch.get("risk_flags")
    if expected_flags is not None and result.patch is not None:
        assert result.patch.get("risk_flags") == expected_flags


def test_eval_cases_success_rate() -> None:
    results = [
        prompt_evaluator.evaluate_prompt(case, chain=None, mode="mock") for case in _EVAL_CASES
    ]
    passed = sum(1 for result in results if result.passed)
    total = len(results)
    success_rate = passed / total if total else 0.0
    assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95%."
