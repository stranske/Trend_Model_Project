"""Shared LLM provider resolution for CLI command owners."""

from __future__ import annotations

import logging
import os
from typing import Any


from trend.mc.viz import TrendCLIError
from trend_analysis.llm import (
    LLMProviderConfig,
)

logger = logging.getLogger(__name__)


def _resolve_llm_provider_config(
    provider: str | None = None,
    *,
    model: str | None = None,
) -> LLMProviderConfig:
    provider_name = (
        provider or os.environ.get("TREND_LLM_PROVIDER") or "openai"
    ).lower()
    supported = {"openai", "anthropic", "ollama"}
    if provider_name not in supported:
        raise TrendCLIError(
            f"Unknown LLM provider '{provider_name}'. Expected one of: {', '.join(sorted(supported))}."
        )
    api_key = os.environ.get("TREND_LLM_API_KEY")
    if not api_key:
        if provider_name == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        elif provider_name == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
    model_name = model or os.environ.get("TREND_LLM_MODEL")
    base_url = os.environ.get("TREND_LLM_BASE_URL")
    organization = os.environ.get("TREND_LLM_ORG")
    max_retries = os.environ.get("TREND_LLM_MAX_RETRIES")
    timeout = os.environ.get("TREND_LLM_TIMEOUT")
    max_retries_value: int | None = None
    timeout_value: float | None = None
    if max_retries:
        try:
            max_retries_value = int(max_retries)
        except ValueError as exc:
            raise TrendCLIError("TREND_LLM_MAX_RETRIES must be an integer") from exc
    if timeout:
        try:
            timeout_value = float(timeout)
        except ValueError as exc:
            raise TrendCLIError("TREND_LLM_TIMEOUT must be a number") from exc
    config_kwargs: dict[str, Any] = {"provider": provider_name}
    if model:
        config_kwargs["model"] = model
    if api_key:
        config_kwargs["api_key"] = api_key
    if base_url:
        config_kwargs["base_url"] = base_url
    if organization:
        config_kwargs["organization"] = organization
    if max_retries_value is not None:
        config_kwargs["max_retries"] = max_retries_value
    if timeout_value is not None:
        config_kwargs["timeout"] = timeout_value
    if model_name:
        config_kwargs["model"] = model_name
    return LLMProviderConfig(**config_kwargs)
