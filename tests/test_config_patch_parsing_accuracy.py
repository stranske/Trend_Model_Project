"""Statistical parsing accuracy tests for ConfigPatchChain."""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("langchain_core")

from langchain_core.runnables import RunnableLambda

from trend_analysis.llm.chain import ConfigPatchChain
from trend_analysis.llm.prompts import build_config_patch_prompt


def _build_llm(response_text: str) -> RunnableLambda:
    def _respond(_prompt_value, **_kwargs) -> str:
        return response_text

    return RunnableLambda(_respond)


def _allowed_schema() -> str:
    schema = {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "object",
                "properties": {
                    "top_n": {"type": "integer"},
                    "weighting": {
                        "type": "object",
                        "properties": {
                            "scheme": {"type": "string"},
                        },
                    },
                },
            }
        },
    }
    return json.dumps(schema, indent=2, ensure_ascii=True)


def _build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index in range(100):
        if index % 2 == 0:
            path = "analysis.top_n"
            value = 5 + index
            summary = f"Set analysis.top_n to {value}."
            instruction = f"Select top {value} funds."
        else:
            path = "analysis.weighting.scheme"
            value = "risk_parity" if index % 4 == 1 else "equal_weight"
            summary = f"Set analysis.weighting.scheme to {value}."
            instruction = f"Use {value.replace('_', ' ')} weighting."
        expected_patch = {
            "operations": [{"op": "set", "path": path, "value": value}],
            "risk_flags": [],
            "summary": summary,
        }
        cases.append(
            {
                "instruction": instruction,
                "current_config": {
                    "analysis": {"top_n": 10, "weighting": {"scheme": "equal_weight"}}
                },
                "expected_patch": expected_patch,
                "llm_response": json.dumps(expected_patch, ensure_ascii=True),
            }
        )
    return cases


def _matches_expected(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    expected_ops = expected["operations"]
    if actual.get("operations") != expected_ops:
        return False
    if actual.get("summary") != expected.get("summary"):
        return False
    return True


def test_parsing_accuracy_sampled_cases() -> None:
    cases = _build_cases()
    assert len(cases) >= 100

    allowed_schema = _allowed_schema()
    passed = 0
    failures: list[str] = []
    for index, case in enumerate(cases):
        llm = _build_llm(case["llm_response"])
        chain = ConfigPatchChain(
            llm=llm,
            prompt_builder=build_config_patch_prompt,
            schema=None,
        )
        patch = chain.run(
            current_config=case["current_config"],
            instruction=case["instruction"],
            allowed_schema=allowed_schema,
        )
        patch_payload = patch.model_dump(mode="python")
        if _matches_expected(patch_payload, case["expected_patch"]):
            passed += 1
        else:
            failures.append(f"case-{index}")

    success_rate = passed / len(cases)
    assert (
        success_rate >= 0.95
    ), f"Structured output parsing success rate {success_rate:.2%} below 95%." + (
        f" Failed cases: {', '.join(failures[:5])}." if failures else ""
    )
