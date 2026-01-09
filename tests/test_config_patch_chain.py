"""Tests for ConfigPatchChain."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("langchain_core")

from langchain_core.runnables import RunnableLambda

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


def test_config_patch_chain_flags_unknown_keys() -> None:
    def _respond(_prompt_value, **_kwargs) -> str:
        payload = {
            "operations": [
                {
                    "op": "set",
                    "path": "analysis.turbo_mode",
                    "value": True,
                }
            ],
            "risk_flags": [],
            "summary": "Enable turbo mode.",
        }
        return json.dumps(payload)

    llm = RunnableLambda(_respond)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={
            "type": "object",
            "properties": {
                "analysis": {"type": "object", "properties": {"top_n": {"type": "integer"}}}
            },
        },
    )

    patch = chain.run(
        current_config={"analysis": {"top_n": 8}},
        instruction="Enable turbo mode.",
    )

    assert patch.needs_review is True


def test_config_patch_chain_retries_on_parse_error() -> None:
    prompts: list[str] = []
    responses = iter(
        [
            "not-json",
            json.dumps(
                {
                    "operations": [
                        {
                            "op": "set",
                            "path": "portfolio.max_weight",
                            "value": 0.25,
                        }
                    ],
                    "risk_flags": [],
                    "summary": "Update max_weight",
                }
            ),
        ]
    )

    def _respond(prompt_value, **_kwargs) -> str:
        messages = prompt_value.to_messages()
        prompts.append(messages[0].content)
        return next(responses)

    llm = RunnableLambda(_respond)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
        retries=1,
    )

    patch = chain.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Set max_weight to 0.25.",
    )

    assert patch.summary == "Update max_weight"
    assert len(prompts) == 2
    assert "PREVIOUS ERROR" in prompts[1]
    assert "ValidationError" in prompts[1]


def test_config_patch_chain_retry_exhausted_raises_error() -> None:
    def _respond(_prompt_value, **_kwargs) -> str:
        return "not-json"

    llm = RunnableLambda(_respond)
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
        retries=1,
    )

    with pytest.raises(ValueError) as excinfo:
        chain.run(
            current_config={"portfolio": {"max_weight": 0.2}},
            instruction="Set max_weight to 0.25.",
        )

    message = str(excinfo.value)
    assert "Failed to parse ConfigPatch after 2 attempts" in message
    assert "ValidationError" in message
