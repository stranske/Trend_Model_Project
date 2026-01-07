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
