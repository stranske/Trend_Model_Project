"""Shared LLM settings helpers for Streamlit components."""

from __future__ import annotations

import logging
import os

import streamlit as st

from trend_analysis.llm import LLMProviderConfig

logger = logging.getLogger(__name__)
_PLACEHOLDER_PREFIXES = ("YOUR_", "CHANGE_ME", "REPLACE_ME")
_ALLOWED_KEY_NAMES = {
    "TS_STREAMLIT_API_KEY",
    "TREND_LLM_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "TS_LLM_PROXY_TOKEN",
}


def sanitize_api_key(value: str | None) -> str | None:
    if not value:
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    upper = trimmed.upper()
    if upper.startswith(_PLACEHOLDER_PREFIXES):
        return None
    return trimmed


def read_secret(key: str) -> str | None:
    try:
        return st.secrets.get(key)
    except (KeyError, FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.debug("Unable to read Streamlit secret %s: %s", key, exc)
        return None


def resolve_api_key_input(raw: str | None) -> str | None:
    if not raw:
        return None
    trimmed = sanitize_api_key(raw)
    if not trimmed:
        return None
    canonical = trimmed.upper()
    if canonical in _ALLOWED_KEY_NAMES:
        secret_val = sanitize_api_key(read_secret(canonical))
        if secret_val:
            return secret_val
        env_val = sanitize_api_key(os.environ.get(canonical))
        if env_val:
            return env_val
    return trimmed


def default_api_key(provider_name: str) -> str | None:
    proxy_url = os.environ.get("TS_LLM_PROXY_URL")
    if proxy_url:
        token = os.environ.get("TS_LLM_PROXY_TOKEN")
        token = sanitize_api_key(token)
        if token:
            return token
    secrets_key = sanitize_api_key(read_secret("TS_STREAMLIT_API_KEY"))
    if secrets_key:
        return secrets_key
    secrets_key = sanitize_api_key(read_secret("TREND_LLM_API_KEY"))
    if secrets_key:
        return secrets_key
    secrets_key = sanitize_api_key(read_secret("OPENAI_API_KEY"))
    if secrets_key:
        return secrets_key
    env_key = sanitize_api_key(os.environ.get("TS_STREAMLIT_API_KEY"))
    if env_key:
        return env_key
    env_key = sanitize_api_key(os.environ.get("TREND_LLM_API_KEY"))
    if env_key:
        return env_key
    if provider_name == "openai":
        return sanitize_api_key(os.environ.get("OPENAI_API_KEY"))
    if provider_name == "anthropic":
        secrets_anthropic = sanitize_api_key(read_secret("ANTHROPIC_API_KEY"))
        if secrets_anthropic:
            return secrets_anthropic
        return sanitize_api_key(os.environ.get("ANTHROPIC_API_KEY"))
    return None


def resolve_llm_provider_config(
    provider: str | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    organization: str | None = None,
) -> LLMProviderConfig:
    provider_name = (provider or os.environ.get("TREND_LLM_PROVIDER") or "openai").lower()
    supported = {"openai", "anthropic", "ollama"}
    if provider_name not in supported:
        raise ValueError(
            f"Unknown LLM provider '{provider_name}'. "
            f"Expected one of: {', '.join(sorted(supported))}."
        )
    resolved_api_key = sanitize_api_key(api_key)
    if not resolved_api_key:
        resolved_api_key = sanitize_api_key(read_secret("TS_STREAMLIT_API_KEY"))
    if not resolved_api_key:
        resolved_api_key = sanitize_api_key(read_secret("OPENAI_API_KEY"))
    if not resolved_api_key:
        resolved_api_key = sanitize_api_key(read_secret("TREND_LLM_API_KEY"))
    if not resolved_api_key:
        resolved_api_key = sanitize_api_key(os.environ.get("TS_STREAMLIT_API_KEY"))
    if not resolved_api_key:
        resolved_api_key = sanitize_api_key(os.environ.get("OPENAI_API_KEY"))
    if not resolved_api_key:
        resolved_api_key = sanitize_api_key(os.environ.get("TREND_LLM_API_KEY"))
    if not resolved_api_key and provider_name == "anthropic":
        resolved_api_key = sanitize_api_key(read_secret("ANTHROPIC_API_KEY"))
    if not resolved_api_key and provider_name == "anthropic":
        resolved_api_key = sanitize_api_key(os.environ.get("ANTHROPIC_API_KEY"))
    if provider_name in {"openai", "anthropic"} and not resolved_api_key:
        env_hint = "OPENAI_API_KEY" if provider_name == "openai" else "ANTHROPIC_API_KEY"
        raise ValueError(
            f"Missing API key for {provider_name}. "
            f"Set TS_STREAMLIT_API_KEY, OPENAI_API_KEY, TREND_LLM_API_KEY, or {env_hint}."
        )
    resolved_model = model or os.environ.get("TREND_LLM_MODEL")
    resolved_base_url = base_url or os.environ.get("TREND_LLM_BASE_URL")
    resolved_org = organization or os.environ.get("TREND_LLM_ORG")
    kwargs: dict[str, object] = {"provider": provider_name}
    if resolved_model:
        kwargs["model"] = resolved_model
    if resolved_api_key:
        kwargs["api_key"] = resolved_api_key
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    if resolved_org:
        kwargs["organization"] = resolved_org
    return LLMProviderConfig(**kwargs)


__all__ = [
    "default_api_key",
    "read_secret",
    "resolve_api_key_input",
    "resolve_llm_provider_config",
    "sanitize_api_key",
]
