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
    "CLAUDE_API_STRANSKE",
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
        secrets = getattr(st, "secrets", None)
        if secrets is None:
            return None
        getter = getattr(secrets, "get", None)
        if getter is None:
            return None
        return getter(key)
    except (
        KeyError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
    ) as exc:
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


def resolve_anthropic_api_key(
    *,
    include_secrets: bool = True,
    include_env: bool = True,
) -> str | None:
    """Resolve Anthropic API key with precedence: CLAUDE_API_STRANSKE, then ANTHROPIC_API_KEY."""
    # Precedence order: CLAUDE_API_STRANSKE (primary), then ANTHROPIC_API_KEY (fallback).
    if include_secrets:
        key = sanitize_api_key(read_secret("CLAUDE_API_STRANSKE"))
        if key:
            return key
    if include_env:
        key = sanitize_api_key(os.environ.get("CLAUDE_API_STRANSKE"))
        if key:
            return key
    if include_secrets:
        key = sanitize_api_key(read_secret("ANTHROPIC_API_KEY"))
        if key:
            return key
    if include_env:
        return sanitize_api_key(os.environ.get("ANTHROPIC_API_KEY"))
    return None


def anthropic_api_key_status(
    *,
    include_secrets: bool = True,
    include_env: bool = True,
) -> dict[str, bool]:
    """Report which Anthropic API key sources are present."""
    status = {"CLAUDE_API_STRANSKE": False, "ANTHROPIC_API_KEY": False}
    if include_secrets:
        if sanitize_api_key(read_secret("CLAUDE_API_STRANSKE")):
            status["CLAUDE_API_STRANSKE"] = True
        if sanitize_api_key(read_secret("ANTHROPIC_API_KEY")):
            status["ANTHROPIC_API_KEY"] = True
    if include_env:
        if sanitize_api_key(os.environ.get("CLAUDE_API_STRANSKE")):
            status["CLAUDE_API_STRANSKE"] = True
        if sanitize_api_key(os.environ.get("ANTHROPIC_API_KEY")):
            status["ANTHROPIC_API_KEY"] = True
    return status


def default_api_key(provider_name: str) -> str | None:
    proxy_url = os.environ.get("TS_LLM_PROXY_URL")
    if proxy_url:
        token = os.environ.get("TS_LLM_PROXY_TOKEN")
        token = sanitize_api_key(token)
        if token:
            return token
    env_key = sanitize_api_key(os.environ.get("TS_STREAMLIT_API_KEY"))
    if env_key:
        return env_key
    env_key = sanitize_api_key(os.environ.get("TREND_LLM_API_KEY"))
    if env_key:
        return env_key
    if provider_name == "openai":
        env_key = sanitize_api_key(os.environ.get("OPENAI_API_KEY"))
        if env_key:
            return env_key
    if provider_name == "anthropic":
        env_key = resolve_anthropic_api_key(include_secrets=False, include_env=True)
        if env_key:
            return env_key
    secrets_key = sanitize_api_key(read_secret("TS_STREAMLIT_API_KEY"))
    if secrets_key:
        return secrets_key
    secrets_key = sanitize_api_key(read_secret("TREND_LLM_API_KEY"))
    if secrets_key:
        return secrets_key
    secrets_key = sanitize_api_key(read_secret("OPENAI_API_KEY"))
    if secrets_key:
        return secrets_key
    if provider_name == "anthropic":
        secrets_anthropic = resolve_anthropic_api_key(include_secrets=True, include_env=False)
        if secrets_anthropic:
            return secrets_anthropic
    return None


def resolve_llm_provider_config(
    provider: str | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    organization: str | None = None,
    require_api_key: bool = True,
) -> LLMProviderConfig:
    provider_name = (provider or os.environ.get("TREND_LLM_PROVIDER") or "openai").lower()
    supported = {"openai", "anthropic", "ollama"}
    if provider_name not in supported:
        raise ValueError(
            f"Unknown LLM provider '{provider_name}'. "
            f"Expected one of: {', '.join(sorted(supported))}."
        )
    resolved_api_key = sanitize_api_key(api_key)
    if not resolved_api_key and provider_name == "openai":
        # Prefer explicit env override for OpenAI (useful for local runs and tests).
        resolved_api_key = sanitize_api_key(os.environ.get("OPENAI_API_KEY"))
    if not resolved_api_key and provider_name == "anthropic":
        # Anthropic resolver already encodes secrets/env precedence rules.
        resolved_api_key = resolve_anthropic_api_key()
    if not resolved_api_key:
        resolved_api_key = sanitize_api_key(os.environ.get("TS_STREAMLIT_API_KEY"))
    if not resolved_api_key:
        resolved_api_key = sanitize_api_key(os.environ.get("TREND_LLM_API_KEY"))
    if not resolved_api_key:
        resolved_api_key = sanitize_api_key(read_secret("TS_STREAMLIT_API_KEY"))
    if not resolved_api_key:
        resolved_api_key = sanitize_api_key(read_secret("TREND_LLM_API_KEY"))
    if not resolved_api_key and provider_name == "openai":
        resolved_api_key = sanitize_api_key(read_secret("OPENAI_API_KEY"))
    if provider_name in {"openai", "anthropic"} and not resolved_api_key and require_api_key:
        env_hint = (
            "OPENAI_API_KEY"
            if provider_name == "openai"
            else "CLAUDE_API_STRANSKE (preferred) or ANTHROPIC_API_KEY"
        )
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
    "anthropic_api_key_status",
    "default_api_key",
    "read_secret",
    "resolve_api_key_input",
    "resolve_anthropic_api_key",
    "resolve_llm_provider_config",
    "sanitize_api_key",
]
