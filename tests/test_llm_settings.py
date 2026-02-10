"""Tests for Streamlit LLM settings resolver behavior."""

from __future__ import annotations

import pytest

from streamlit_app.components.llm_settings import (
    resolve_anthropic_api_key,
    resolve_llm_provider_config,
)


_ENV_KEYS = (
    "CLAUDE_API_STRANSKE",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "TREND_LLM_API_KEY",
    "TS_STREAMLIT_API_KEY",
)


@pytest.fixture(autouse=True)
def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_resolve_anthropic_key_primary_env() -> None:
    value = "claude-primary"
    assert resolve_anthropic_api_key() is None
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CLAUDE_API_STRANSKE", value)
        assert resolve_anthropic_api_key() == value


def test_anthropic_provider_config_from_primary_env() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CLAUDE_API_STRANSKE", "claude-primary")
        config = resolve_llm_provider_config("anthropic")
        assert config.provider == "anthropic"
        assert config.api_key == "claude-primary"


def test_anthropic_provider_config_no_error_with_primary_env() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CLAUDE_API_STRANSKE", "claude-primary")
        resolve_llm_provider_config("anthropic")


def test_resolve_anthropic_key_fallback_env() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ANTHROPIC_API_KEY", "anthropic-fallback")
        assert resolve_anthropic_api_key() == "anthropic-fallback"


def test_anthropic_key_precedence_primary_over_fallback() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CLAUDE_API_STRANSKE", "claude-primary")
        mp.setenv("ANTHROPIC_API_KEY", "anthropic-fallback")
        assert resolve_anthropic_api_key() == "claude-primary"


def test_anthropic_key_missing_returns_none_and_config_errors() -> None:
    assert resolve_anthropic_api_key() is None
    with pytest.raises(ValueError):
        resolve_llm_provider_config("anthropic")
