"""Integration tests for ConfigPatchChain eval cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("langchain_core")

from langchain_core.runnables import RunnableLambda

from trend_analysis.llm.chain import ConfigPatchChain
from trend_analysis.llm.prompts import build_config_patch_prompt

_EVAL_CASES_PATH = Path("tools/eval_test_cases.yml")


def _load_case(case_id: str) -> dict[str, Any]:
    payload = yaml.safe_load(_EVAL_CASES_PATH.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    for case in cases:
        if case.get("id") == case_id:
            return case
    raise KeyError(f"Eval case not found: {case_id}")


def _serialize_schema(schema: Any | None) -> str | None:
    if schema is None:
        return None
    if isinstance(schema, str):
        return schema
    return json.dumps(schema, indent=2, ensure_ascii=True)


def _build_chain(response_text: str, schema: dict[str, Any] | None) -> ConfigPatchChain:
    def _respond(_prompt_value, **_kwargs) -> str:
        return response_text

    llm = RunnableLambda(_respond)
    return ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema=schema,
    )


def test_risk_parity_weighting_case() -> None:
    case = _load_case("risk_parity_weighting")
    expected = case["expected_patch"]
    response_text = json.dumps(expected, ensure_ascii=True)
    chain = _build_chain(response_text, schema=case.get("schema"))

    patch = chain.run(
        current_config=case["current_config"],
        instruction=case["instruction"],
        allowed_schema=_serialize_schema(case.get("allowed_schema")),
        system_prompt=case.get("system_prompt"),
        safety_rules=case.get("safety_rules"),
    )

    assert len(patch.operations) == 1
    assert patch.operations[0].path == "analysis.weighting.scheme"
    assert patch.operations[0].value == "risk_parity"
    assert patch.summary == expected["summary"]
