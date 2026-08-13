"""Provider factory for LLM integrations."""

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Literal
from urllib.parse import urlsplit


@dataclass(slots=True)
class LLMProviderConfig:
    """Configuration for selecting and instantiating an LLM provider."""

    provider: Literal["openai", "anthropic", "ollama"]
    model: str = "gpt-4o-mini"
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    organization: str | None = None
    timeout: float | None = None
    max_retries: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BrowserOpenAICompatibleChat:
    """Minimal callable adapter for a CORS-enabled chat-completions endpoint.

    Provider SDKs currently depend on native wheels that Pyodide cannot load.
    This adapter preserves the LangChain prompt pipeline while keeping the
    browser transport explicit and limited to the endpoint selected by the
    user at runtime.
    """

    base_url: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    organization: str | None = None
    timeout: float | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def bind(self, **kwargs: Any) -> "BrowserOpenAICompatibleChat":
        """Return a bound adapter, matching the subset LangChain chains use."""

        known = {
            name: kwargs.pop(name)
            for name in tuple(kwargs)
            if name
            in {
                "model",
                "temperature",
                "max_tokens",
            }
        }
        return replace(self, extra={**self.extra, **kwargs}, **known)

    def __call__(self, prompt: Any) -> str:
        return self.invoke(prompt)

    def invoke(self, prompt: Any, *_args: Any, **_kwargs: Any) -> str:
        """Send one OpenAI-compatible request through Pyodide's HTTP bridge."""

        try:
            from pyodide_http import patch as patch_http
        except ImportError as exc:  # pragma: no cover - only reachable in browser packaging
            raise RuntimeError("The browser HTTP bridge is unavailable.") from exc

        try:
            import requests
        except ImportError as exc:  # pragma: no cover - supplied by the Pyodide runtime
            raise RuntimeError("The browser HTTP client is unavailable.") from exc

        patch_http()
        prompt_text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt_text}],
            **self.extra,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        request_kwargs: dict[str, Any] = {"headers": headers, "json": payload}
        if self.timeout is not None:
            request_kwargs["timeout"] = self.timeout
        response = requests.post(_chat_completions_url(self.base_url), **request_kwargs)
        response.raise_for_status()
        try:
            response_payload = response.json()
            content = response_payload["choices"][0]["message"]["content"]
        except (AttributeError, IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "The configured endpoint returned an invalid chat response."
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("The configured endpoint returned an empty chat response.")
        return content


def create_llm(config: LLMProviderConfig) -> Any:
    """Instantiate a LangChain LLM based on the provider configuration."""

    provider = config.provider.lower()
    if _running_in_pyodide():
        if provider != "openai":
            raise RuntimeError(
                "The browser demo supports OpenAI-compatible endpoints only; "
                "select OpenAI-compatible and enter a CORS-enabled base URL."
            )
        if not config.base_url:
            raise RuntimeError(
                "The browser demo requires an explicit CORS-enabled OpenAI-compatible base URL."
            )
        return BrowserOpenAICompatibleChat(
            base_url=config.base_url,
            model=config.model,
            api_key=_resolve_api_key(provider, config.api_key),
            organization=config.organization,
            timeout=config.timeout,
            extra=dict(config.extra),
        )
    if provider == "openai":
        return _create_provider("langchain_openai", "ChatOpenAI", config, provider=provider)
    if provider == "anthropic":
        return _create_provider("langchain_anthropic", "ChatAnthropic", config, provider=provider)
    if provider == "ollama":
        return _create_provider("langchain_ollama", "ChatOllama", config, provider=provider)
    raise ValueError(f"Unknown provider: {config.provider}")


def _create_provider(
    module_name: str,
    class_name: str,
    config: LLMProviderConfig,
    *,
    provider: str,
) -> Any:
    provider_cls = _import_provider(module_name, class_name)
    return provider_cls(**_build_kwargs(config, provider=provider))


def _import_provider(module_name: str, class_name: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Provider dependency '{module_name}' is not installed. "
            "Install the Trend Model LLM extras to use this provider."
        ) from exc
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise RuntimeError(f"Provider class '{class_name}' not found in '{module_name}'.") from exc


def _build_kwargs(config: LLMProviderConfig, *, provider: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": config.model}
    api_key = _resolve_api_key(provider, config.api_key)
    if api_key:
        kwargs["api_key"] = api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.organization:
        kwargs["organization"] = config.organization
    if config.timeout is not None:
        kwargs["timeout"] = config.timeout
    if config.max_retries is not None:
        kwargs["max_retries"] = config.max_retries
    if config.extra:
        kwargs.update(config.extra)
    return kwargs


def _resolve_api_key(provider: str, api_key: str | None) -> str | None:
    if api_key:
        return api_key
    env_override = os.environ.get("TREND_LLM_API_KEY")
    if env_override:
        return env_override
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY")
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    return None


def _running_in_pyodide() -> bool:
    return sys.platform == "emscripten"


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("The OpenAI-compatible base URL must be an absolute HTTP(S) URL.")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


__all__ = ["BrowserOpenAICompatibleChat", "LLMProviderConfig", "create_llm"]
