"""Tests for LLM provider factory."""

from __future__ import annotations

import sys
import types

import pytest

from trend_analysis.llm.providers import LLMProviderConfig, create_llm


@pytest.mark.parametrize(
    ("provider", "module_name", "class_name"),
    [
        ("openai", "langchain_openai", "ChatOpenAI"),
        ("anthropic", "langchain_anthropic", "ChatAnthropic"),
    ],
)
def test_create_llm_instantiates_provider(
    provider: str,
    module_name: str,
    class_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType(module_name)
    created: dict[str, dict[str, object]] = {}

    class Dummy:
        def __init__(self, **kwargs) -> None:
            created["kwargs"] = kwargs

    setattr(module, class_name, Dummy)
    monkeypatch.setitem(sys.modules, module_name, module)

    config = LLMProviderConfig(
        provider=provider,  # type: ignore[arg-type]
        model="unit-test-model",
        api_key="unit-test-key",
        base_url="https://example.invalid",
        timeout=3.5,
        max_retries=4,
        extra={"request_timeout": 7},
    )

    llm = create_llm(config)

    assert isinstance(llm, Dummy)
    assert created["kwargs"]["model"] == "unit-test-model"
    assert created["kwargs"]["api_key"] == "unit-test-key"
    assert created["kwargs"]["base_url"] == "https://example.invalid"
    assert created["kwargs"]["timeout"] == 3.5
    assert created["kwargs"]["max_retries"] == 4
    assert created["kwargs"]["request_timeout"] == 7


def test_create_llm_unknown_provider_raises() -> None:
    config = LLMProviderConfig(provider="openai")

    config.provider = "unknown"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unknown provider"):
        create_llm(config)
