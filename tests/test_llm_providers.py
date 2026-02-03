"""Unit tests for LLM provider factory behavior."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from trend_analysis.llm.providers import LLMProviderConfig, create_llm


class DummyProvider:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)


def _register_provider(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str,
) -> None:
    module = types.ModuleType(module_name)
    setattr(module, class_name, DummyProvider)
    monkeypatch.setitem(sys.modules, module_name, module)


@pytest.mark.parametrize(
    ("provider", "module_name", "class_name"),
    [
        ("openai", "langchain_openai", "ChatOpenAI"),
        ("anthropic", "langchain_anthropic", "ChatAnthropic"),
        ("ollama", "langchain_ollama", "ChatOllama"),
    ],
)
def test_create_llm_selects_provider_class(
    provider: str,
    module_name: str,
    class_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_provider(monkeypatch, module_name, class_name)

    config = LLMProviderConfig(provider=provider, model="unit-test-model")  # type: ignore[arg-type]
    llm = create_llm(config)

    assert isinstance(llm, DummyProvider)
    assert llm.kwargs["model"] == "unit-test-model"


def test_openai_api_key_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    _register_provider(monkeypatch, "langchain_openai", "ChatOpenAI")

    monkeypatch.delenv("TREND_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    explicit = LLMProviderConfig(provider="openai", model="unit", api_key="explicit")
    llm = create_llm(explicit)
    assert llm.kwargs.get("api_key") == "explicit"

    monkeypatch.delenv("TREND_LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env")
    llm = create_llm(LLMProviderConfig(provider="openai", model="unit"))
    assert llm.kwargs.get("api_key") == "openai-env"

    monkeypatch.setenv("TREND_LLM_API_KEY", "override")
    llm = create_llm(LLMProviderConfig(provider="openai", model="unit"))
    assert llm.kwargs.get("api_key") == "override"


def test_anthropic_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _register_provider(monkeypatch, "langchain_anthropic", "ChatAnthropic")

    monkeypatch.delenv("TREND_LLM_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-env")
    llm = create_llm(LLMProviderConfig(provider="anthropic", model="unit"))
    assert llm.kwargs.get("api_key") == "anthropic-env"


def test_ollama_does_not_require_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _register_provider(monkeypatch, "langchain_ollama", "ChatOllama")

    monkeypatch.delenv("TREND_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    config = LLMProviderConfig(provider="ollama", model="llama3", base_url="http://localhost:11434")
    llm = create_llm(config)

    assert llm.kwargs["model"] == "llama3"
    assert llm.kwargs["base_url"] == "http://localhost:11434"
    assert "api_key" not in llm.kwargs
