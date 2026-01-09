"""Provider factory for LLM integrations."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class LLMProviderConfig:
    """Configuration for selecting and instantiating an LLM provider."""

    provider: Literal["openai", "anthropic", "ollama"]
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    organization: str | None = None
    timeout: float | None = None
    max_retries: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def create_llm(config: LLMProviderConfig) -> Any:
    """Instantiate a LangChain LLM based on the provider configuration."""

    provider = config.provider.lower()
    if provider == "openai":
        return _create_provider("langchain_openai", "ChatOpenAI", config)
    if provider == "anthropic":
        return _create_provider("langchain_anthropic", "ChatAnthropic", config)
    if provider == "ollama":
        return _create_provider("langchain_ollama", "ChatOllama", config)
    raise ValueError(f"Unknown provider: {config.provider}")


def _create_provider(module_name: str, class_name: str, config: LLMProviderConfig) -> Any:
    provider_cls = _import_provider(module_name, class_name)
    return provider_cls(**_build_kwargs(config))


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


def _build_kwargs(config: LLMProviderConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": config.model}
    if config.api_key:
        kwargs["api_key"] = config.api_key
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


__all__ = ["LLMProviderConfig", "create_llm"]
