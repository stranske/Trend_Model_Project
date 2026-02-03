"""Tests for LLM provider factory."""

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
) -> DummyProvider:
    module = types.ModuleType(module_name)
    setattr(module, class_name, DummyProvider)
    monkeypatch.setitem(sys.modules, module_name, module)
    return DummyProvider


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
    _register_provider(monkeypatch, module_name, class_name)

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

    assert isinstance(llm, DummyProvider)
    assert llm.kwargs["model"] == "unit-test-model"
    assert llm.kwargs["api_key"] == "unit-test-key"
    assert llm.kwargs["base_url"] == "https://example.invalid"
    assert llm.kwargs["timeout"] == 3.5
    assert llm.kwargs["max_retries"] == 4
    assert llm.kwargs["request_timeout"] == 7


def test_create_llm_instantiates_ollama_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _register_provider(monkeypatch, "langchain_ollama", "ChatOllama")

    config = LLMProviderConfig(
        provider="ollama",
        model="unit-test-model",
        base_url="http://example.invalid",
        extra={"temperature": 0.05},
    )

    llm = create_llm(config)

    assert isinstance(llm, DummyProvider)
    assert llm.kwargs["model"] == "unit-test-model"
    assert llm.kwargs["base_url"] == "http://example.invalid"
    assert llm.kwargs["temperature"] == 0.05
    assert "api_key" not in llm.kwargs


@pytest.mark.parametrize(
    ("provider", "env_key"),
    [
        ("openai", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
    ],
)
def test_create_llm_uses_provider_env_api_key(
    provider: str,
    env_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "langchain_openai" if provider == "openai" else "langchain_anthropic"
    class_name = "ChatOpenAI" if provider == "openai" else "ChatAnthropic"
    _register_provider(monkeypatch, module_name, class_name)

    monkeypatch.delenv("TREND_LLM_API_KEY", raising=False)
    monkeypatch.setenv(env_key, "provider-env-key")

    config = LLMProviderConfig(provider=provider, model="unit-test-model")

    llm = create_llm(config)

    assert llm.kwargs["api_key"] == "provider-env-key"


def test_create_llm_uses_trend_env_api_key_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_provider(monkeypatch, "langchain_openai", "ChatOpenAI")

    monkeypatch.setenv("TREND_LLM_API_KEY", "override-key")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")

    config = LLMProviderConfig(provider="openai", model="unit-test-model")

    llm = create_llm(config)

    assert llm.kwargs["api_key"] == "override-key"


def test_create_llm_prefers_explicit_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _register_provider(monkeypatch, "langchain_openai", "ChatOpenAI")

    monkeypatch.setenv("TREND_LLM_API_KEY", "override-key")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")

    config = LLMProviderConfig(provider="openai", model="unit-test-model", api_key="explicit")

    llm = create_llm(config)

    assert llm.kwargs["api_key"] == "explicit"


def test_create_llm_unknown_provider_raises() -> None:
    config = LLMProviderConfig(provider="openai")

    config.provider = "unknown"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unknown provider"):
        create_llm(config)
