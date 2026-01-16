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


def test_create_llm_instantiates_ollama_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "langchain_ollama"
    class_name = "ChatOllama"
    module = types.ModuleType(module_name)
    created: dict[str, dict[str, object]] = {}

    class Dummy:
        def __init__(self, **kwargs) -> None:
            created["kwargs"] = kwargs

    setattr(module, class_name, Dummy)
    monkeypatch.setitem(sys.modules, module_name, module)

    config = LLMProviderConfig(
        provider="ollama",
        model="unit-test-model",
        base_url="http://example.invalid",
        extra={"temperature": 0.05},
    )

    llm = create_llm(config)

    assert isinstance(llm, Dummy)
    assert created["kwargs"]["model"] == "unit-test-model"
    assert created["kwargs"]["base_url"] == "http://example.invalid"
    assert created["kwargs"]["temperature"] == 0.05
    assert "api_key" not in created["kwargs"]


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
    module = types.ModuleType(module_name)
    created: dict[str, dict[str, object]] = {}

    class Dummy:
        def __init__(self, **kwargs) -> None:
            created["kwargs"] = kwargs

    setattr(module, class_name, Dummy)
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.delenv("TREND_LLM_API_KEY", raising=False)
    monkeypatch.setenv(env_key, "provider-env-key")

    config = LLMProviderConfig(provider=provider, model="unit-test-model")

    create_llm(config)

    assert created["kwargs"]["api_key"] == "provider-env-key"


def test_create_llm_uses_trend_env_api_key_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "langchain_openai"
    class_name = "ChatOpenAI"
    module = types.ModuleType(module_name)
    created: dict[str, dict[str, object]] = {}

    class Dummy:
        def __init__(self, **kwargs) -> None:
            created["kwargs"] = kwargs

    setattr(module, class_name, Dummy)
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setenv("TREND_LLM_API_KEY", "override-key")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")

    config = LLMProviderConfig(provider="openai", model="unit-test-model")

    create_llm(config)

    assert created["kwargs"]["api_key"] == "override-key"


def test_create_llm_unknown_provider_raises() -> None:
    config = LLMProviderConfig(provider="openai")

    config.provider = "unknown"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unknown provider"):
        create_llm(config)
