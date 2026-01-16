"""Integration tests for ConfigPatchChain using LLM providers."""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

pytest.importorskip("langchain_core")

from langchain_core.runnables import RunnableLambda

from trend_analysis.llm.chain import ConfigPatchChain
from trend_analysis.llm.prompts import build_config_patch_prompt
from trend_analysis.llm.providers import LLMProviderConfig, create_llm


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content

    def __str__(self) -> str:
        return self.content


def _register_provider(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str,
) -> None:
    module = types.ModuleType(module_name)

    class DummyProvider(RunnableLambda):
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = dict(kwargs)
            self._response_text = self.kwargs.pop("response_text")
            self._return_mode = self.kwargs.pop("return_mode", "content")
            super().__init__(self._respond)

        def _respond(self, _input: dict[str, Any], **_kwargs: Any) -> Any:
            if self._return_mode == "content":
                return _Message(self._response_text)
            return self._response_text

    setattr(module, class_name, DummyProvider)
    monkeypatch.setitem(sys.modules, module_name, module)


@pytest.mark.parametrize(
    ("provider", "module_name", "class_name", "return_mode", "fence_output"),
    [
        ("openai", "langchain_openai", "ChatOpenAI", "content", False),
        ("anthropic", "langchain_anthropic", "ChatAnthropic", "text", True),
    ],
)
def test_chain_runs_with_provider_responses(
    provider: str,
    module_name: str,
    class_name: str,
    return_mode: str,
    fence_output: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_provider(monkeypatch, module_name, class_name)

    payload = {
        "operations": [{"op": "set", "path": "analysis.weighting.scheme", "value": "risk_parity"}],
        "risk_flags": [],
        "summary": "Switch weighting to risk parity.",
    }
    response_text = json.dumps(payload, ensure_ascii=True)
    if fence_output:
        response_text = f"```json\n{response_text}\n```"

    config = LLMProviderConfig(
        provider=provider,  # type: ignore[arg-type]
        model="unit-test-model",
        extra={"response_text": response_text, "return_mode": return_mode},
    )
    llm = create_llm(config)

    schema = {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "object",
                "properties": {
                    "weighting": {
                        "type": "object",
                        "properties": {"scheme": {"type": "string"}},
                    }
                },
            }
        },
    }
    chain = ConfigPatchChain(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema=schema,
    )

    patch = chain.run(
        current_config={"analysis": {"weighting": {"scheme": "equal_weight"}}},
        instruction="Use risk parity weighting.",
    )

    assert patch.summary == payload["summary"]
    assert patch.operations[0].path == "analysis.weighting.scheme"
    assert patch.operations[0].value == "risk_parity"
    assert patch.needs_review is False
