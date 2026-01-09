"""Tests for ConfigPatchChain environment settings."""

from __future__ import annotations

from trend_analysis.llm.chain import ConfigPatchChain
from trend_analysis.llm.prompts import build_config_patch_prompt


class DummyLLM:
    def __init__(self) -> None:
        self.bound: dict[str, object] | None = None

    def bind(self, **kwargs):
        self.bound = kwargs
        return self


def test_chain_from_env_uses_temperature_and_model(monkeypatch) -> None:
    monkeypatch.setenv("TREND_LLM_TEMPERATURE", "0.42")
    monkeypatch.setenv("TREND_LLM_MODEL", "unit-test-model")

    llm = DummyLLM()
    chain = ConfigPatchChain.from_env(
        llm=llm,
        prompt_builder=build_config_patch_prompt,
        schema={"type": "object"},
    )

    assert chain.temperature == 0.42
    assert chain.model == "unit-test-model"

    chain._bind_llm()
    assert llm.bound == {"temperature": 0.42, "model": "unit-test-model"}
