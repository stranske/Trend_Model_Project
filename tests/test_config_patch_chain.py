"""Tests for ConfigPatchChain."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("langchain_core")

import jsonschema
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from trend_analysis.config.patch import ConfigPatch
from trend_analysis.llm.chain import ConfigPatchChain
from trend_analysis.llm.prompts import build_config_patch_prompt


def test_config_patch_chain_run_parses_patch() -> None:
    captured: dict[str, str] = {}

    def _respond(prompt_value, **_kwargs) -> str:
        messages = prompt_value.to_messages()
        captured["prompt"] = messages[0].content
        payload = {
            "operations": [
                {
                    "op": "set",
                    "path": "portfolio.max_weight",
                    "value": 0.3,
                    "rationale": "Align with instruction",
                }
            ],
            "risk_flags": [],
            "summary": "Update max_weight",
        }
        return json.dumps(payload)

    llm = RunnableLambda(_respond)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
    )

    patch = chain.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Set max_weight to 0.3.",
    )

    assert patch.operations[0].path == "portfolio.max_weight"
    assert patch.summary == "Update max_weight"
    assert "USER INSTRUCTION" in captured["prompt"]


def test_config_patch_chain_batch_schema_conformance() -> None:
    total_cases = 100
    valid_cases = 96
    responses: list[str] = []
    for idx in range(total_cases):
        if idx < valid_cases:
            payload = {
                "operations": [
                    {
                        "op": "set",
                        "path": "portfolio.max_weight",
                        "value": round(0.2 + idx * 0.001, 3),
                        "rationale": "Align with instruction",
                    }
                ],
                "risk_flags": [],
                "summary": f"Update max_weight {idx}",
            }
            responses.append(json.dumps(payload))
        else:
            responses.append(json.dumps({"operations": [], "risk_flags": []}))

    response_iter = iter(responses)

    def _respond(prompt_value, **_kwargs) -> str:
        _ = prompt_value.to_messages()
        return next(response_iter)

    llm = RunnableLambda(_respond)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
    )

    schema = ConfigPatch.json_schema()
    validator = jsonschema.Draft202012Validator(schema)
    valid_count = 0
    for idx in range(total_cases):
        prompt = chain.build_prompt(
            current_config={"portfolio": {"max_weight": 0.2}},
            instruction=f"Set max_weight to {0.2 + idx * 0.001:.3f}.",
        )
        response_text = chain._invoke_llm(prompt)
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            continue
        if validator.is_valid(payload):
            valid_count += 1

    assert valid_count / total_cases >= 0.95


def test_config_patch_chain_unknown_keys_raise() -> None:
    def _respond(prompt_value, **_kwargs) -> str:
        _ = prompt_value.to_messages()
        payload = {
            "operations": [],
            "risk_flags": [],
            "summary": "No changes",
            "extra_field": "unexpected",
        }
        return json.dumps(payload)

    llm = RunnableLambda(_respond)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
    )

    with pytest.raises(ValidationError):
        chain.run(
            current_config={"portfolio": {"max_weight": 0.2}},
            instruction="Do nothing.",
        )


def test_config_patch_chain_omits_unknown_key_instruction() -> None:
    def _respond(prompt_value, **_kwargs) -> str:
        _ = prompt_value.to_messages()
        payload = {
            "operations": [],
            "risk_flags": [],
            "summary": "Unknown key portfolio.unknown_setting; no changes applied.",
        }
        return json.dumps(payload)

    llm = RunnableLambda(_respond)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
    )

    patch = chain.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Set portfolio.unknown_setting to 0.5.",
    )

    assert patch.operations == []
    assert "unknown key" in patch.summary.lower()
