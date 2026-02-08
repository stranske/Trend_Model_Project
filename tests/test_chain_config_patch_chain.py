"""Tests for ConfigPatchChain retry prompts."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("langchain_core")

from langchain_core.runnables import RunnableLambda  # noqa: E402

from trend_analysis.llm.chain import ConfigPatchChain  # noqa: E402
from trend_analysis.llm.prompts import build_config_patch_prompt  # noqa: E402


def _extract_prompt_text(prompt_value: object) -> str:
    if hasattr(prompt_value, "to_messages"):
        messages = prompt_value.to_messages()
    else:
        messages = getattr(prompt_value, "messages", [])
    if messages:
        return messages[0].content
    return str(prompt_value)


class _PromptCaptureLLM(RunnableLambda):
    def __init__(self, *, responses: list[str]) -> None:
        self.prompts: list[str] = []
        self._responses = iter(responses)
        super().__init__(self._respond)

    def _respond(self, prompt_value, **_kwargs) -> str:
        self.prompts.append(_extract_prompt_text(prompt_value))
        return next(self._responses)

    def supports_structured_output(self) -> bool:
        return False


def test_retry_prompt_uses_configpatch_schema_note() -> None:
    payload = {
        "operations": [],
        "risk_flags": [],
        "summary": "Ok",
    }
    llm = _PromptCaptureLLM(responses=["not json", json.dumps(payload)])
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
    )

    chain.run(
        current_config={"portfolio": {"max_weight": 0.2}},
        instruction="Set max_weight to 0.4.",
    )

    assert len(llm.prompts) == 2
    assert "ConfigPatch schema" in llm.prompts[1]
