"""Unit tests for LLM provider factory configuration."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from trend_analysis.llm import providers as providers_module
from trend_analysis.llm.providers import (
    BrowserOpenAICompatibleChat,
    LLMProviderConfig,
    create_llm,
)


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


def test_create_llm_uses_browser_adapter_without_provider_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(providers_module, "_running_in_pyodide", lambda: True)

    llm = create_llm(
        LLMProviderConfig(
            provider="openai",
            model="browser-model",
            api_key="session-secret",
            base_url="https://llm.example.test/v1",
        )
    )

    assert isinstance(llm, BrowserOpenAICompatibleChat)
    assert llm.model == "browser-model"
    assert llm.base_url == "https://llm.example.test/v1"
    assert "session-secret" not in repr(llm)


@pytest.mark.parametrize("provider", ["anthropic", "ollama"])
def test_create_llm_rejects_non_openai_browser_providers(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    monkeypatch.setattr(providers_module, "_running_in_pyodide", lambda: True)

    with pytest.raises(RuntimeError, match="OpenAI-compatible endpoints only"):
        create_llm(
            LLMProviderConfig(provider=provider, base_url="https://llm.example.test/v1")  # type: ignore[arg-type]
        )


def test_create_llm_requires_explicit_browser_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers_module, "_running_in_pyodide", lambda: True)

    with pytest.raises(RuntimeError, match="explicit CORS-enabled"):
        create_llm(LLMProviderConfig(provider="openai"))


def test_browser_adapter_posts_only_to_configured_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    patch_calls: list[bool] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "response text"}}]}

    requests_module = types.ModuleType("requests")

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append((url, kwargs))
        return FakeResponse()

    requests_module.post = fake_post  # type: ignore[attr-defined]
    pyodide_http_module = types.ModuleType("pyodide_http")
    pyodide_http_module.patch_requests = lambda: patch_calls.append(True)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", requests_module)
    monkeypatch.setitem(sys.modules, "pyodide_http", pyodide_http_module)

    adapter = BrowserOpenAICompatibleChat(
        base_url="https://llm.example.test/v1/",
        model="browser-model",
        api_key="session-secret",
        organization="test-org",
        timeout=12.0,
    ).bind(temperature=0.25, max_tokens=321)

    assert adapter.invoke("hello") == "response text"
    assert patch_calls == [True]
    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url == "https://llm.example.test/v1/chat/completions"
    assert kwargs["headers"] == {
        "Content-Type": "application/json",
        "Authorization": "Bearer session-secret",
        "OpenAI-Organization": "test-org",
    }
    assert kwargs["json"] == {
        "model": "browser-model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.25,
        "max_tokens": 321,
    }
    assert kwargs["timeout"] == 12.0


def test_browser_adapter_preserves_chat_prompt_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "ok"}}]}

    requests_module = types.ModuleType("requests")
    requests_module.post = lambda _url, **kwargs: (  # type: ignore[attr-defined]
        calls.append(kwargs) or FakeResponse()
    )
    pyodide_http_module = types.ModuleType("pyodide_http")
    pyodide_http_module.patch_requests = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", requests_module)
    monkeypatch.setitem(sys.modules, "pyodide_http", pyodide_http_module)

    class FakePrompt:
        def to_messages(self) -> list[Any]:
            return [
                types.SimpleNamespace(type="system", content="Follow policy", additional_kwargs={}),
                types.SimpleNamespace(
                    type="human", content="Explain results", additional_kwargs={}
                ),
            ]

    adapter = BrowserOpenAICompatibleChat(
        base_url="https://llm.example.test/v1", model="browser-model"
    )
    assert adapter.invoke(FakePrompt()) == "ok"
    assert calls[0]["json"]["messages"] == [
        {"role": "system", "content": "Follow policy"},
        {"role": "user", "content": "Explain results"},
    ]
