"""Unit tests for LLM provider factory configuration."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from trend_analysis.llm.providers import LLMProviderConfig, create_llm


def _register_provider(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str,
) -> type:
    module = types.ModuleType(module_name)

    class DummyProvider:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = dict(kwargs)

    setattr(module, class_name, DummyProvider)
    monkeypatch.setitem(sys.modules, module_name, module)
    return DummyProvider


def test_create_llm_openai_uses_explicit_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_cls = _register_provider(monkeypatch, "langchain_openai", "ChatOpenAI")
    config = LLMProviderConfig(provider="openai", model="unit-test-model", api_key="key-123")

    llm = create_llm(config)

    assert isinstance(llm, provider_cls)
    assert llm.kwargs["model"] == "unit-test-model"
    assert llm.kwargs["api_key"] == "key-123"


def test_create_llm_openai_uses_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_cls = _register_provider(monkeypatch, "langchain_openai", "ChatOpenAI")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env-key")
    monkeypatch.delenv("TREND_LLM_API_KEY", raising=False)

    config = LLMProviderConfig(provider="openai", model="unit-test-model")
    llm = create_llm(config)

    assert isinstance(llm, provider_cls)
    assert llm.kwargs["api_key"] == "openai-env-key"


def test_create_llm_openai_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_cls = _register_provider(monkeypatch, "langchain_openai", "ChatOpenAI")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TREND_LLM_API_KEY", raising=False)

    config = LLMProviderConfig(provider="openai", model="unit-test-model")
    llm = create_llm(config)

    assert isinstance(llm, provider_cls)
    assert "api_key" not in llm.kwargs


def test_create_llm_anthropic_with_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_cls = _register_provider(monkeypatch, "langchain_anthropic", "ChatAnthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-env-key")
    monkeypatch.delenv("TREND_LLM_API_KEY", raising=False)

    config = LLMProviderConfig(provider="anthropic", model="unit-test-model")
    llm = create_llm(config)

    assert isinstance(llm, provider_cls)
    assert llm.kwargs["api_key"] == "anthropic-env-key"


def test_create_llm_anthropic_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_cls = _register_provider(monkeypatch, "langchain_anthropic", "ChatAnthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TREND_LLM_API_KEY", raising=False)

    config = LLMProviderConfig(provider="anthropic", model="unit-test-model")
    llm = create_llm(config)

    assert isinstance(llm, provider_cls)
    assert "api_key" not in llm.kwargs


def test_create_llm_ollama_supports_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_cls = _register_provider(monkeypatch, "langchain_ollama", "ChatOllama")
    monkeypatch.delenv("TREND_LLM_API_KEY", raising=False)

    config = LLMProviderConfig(
        provider="ollama",
        model="unit-test-model",
        base_url="http://localhost:11434",
    )
    llm = create_llm(config)

    assert isinstance(llm, provider_cls)
    assert llm.kwargs["model"] == "unit-test-model"
    assert llm.kwargs["base_url"] == "http://localhost:11434"
    assert "api_key" not in llm.kwargs


def test_create_llm_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        create_llm(LLMProviderConfig(provider="unknown", model="unit-test-model"))  # type: ignore[arg-type]
