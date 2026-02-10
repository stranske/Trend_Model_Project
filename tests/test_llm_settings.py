"""Tests for Streamlit LLM settings resolver behavior."""

from __future__ import annotations

import pytest

from streamlit_app.components.llm_settings import (
    anthropic_api_key_status,
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


def test_anthropic_provider_config_from_fallback_env() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ANTHROPIC_API_KEY", "anthropic-fallback")
        config = resolve_llm_provider_config("anthropic")
        assert config.provider == "anthropic"
        assert config.api_key == "anthropic-fallback"


def test_anthropic_provider_config_no_error_with_fallback_env() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ANTHROPIC_API_KEY", "anthropic-fallback")
        resolve_llm_provider_config("anthropic")


def test_anthropic_key_precedence_primary_over_fallback() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CLAUDE_API_STRANSKE", "claude-primary")
        mp.setenv("ANTHROPIC_API_KEY", "anthropic-fallback")
        assert resolve_anthropic_api_key() == "claude-primary"


def test_anthropic_provider_config_prefers_primary_over_fallback() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CLAUDE_API_STRANSKE", "claude-primary")
        mp.setenv("ANTHROPIC_API_KEY", "anthropic-fallback")
        config = resolve_llm_provider_config("anthropic")
        assert config.api_key == "claude-primary"


def test_anthropic_key_missing_returns_none_and_config_errors() -> None:
    assert resolve_anthropic_api_key() is None
    with pytest.raises(ValueError):
        resolve_llm_provider_config("anthropic")


def test_anthropic_key_status_reports_primary_env() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CLAUDE_API_STRANSKE", "claude-primary")
        status = anthropic_api_key_status(include_secrets=False, include_env=True)
        assert status == {"CLAUDE_API_STRANSKE": True, "ANTHROPIC_API_KEY": False}


# --- Secrets-based resolution path tests ---


def test_resolve_anthropic_key_secret_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Secrets path: CLAUDE_API_STRANSKE secret is preferred over env fallback."""
    import streamlit as st

    fake_secrets = {"CLAUDE_API_STRANSKE": "secret-primary"}
    monkeypatch.setattr(st, "secrets", fake_secrets)
    result = resolve_anthropic_api_key(include_secrets=True, include_env=True)
    assert result == "secret-primary"


def test_resolve_anthropic_key_secret_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Secrets path: ANTHROPIC_API_KEY secret used when CLAUDE_API_STRANSKE absent."""
    import streamlit as st

    fake_secrets = {"ANTHROPIC_API_KEY": "secret-fallback"}
    monkeypatch.setattr(st, "secrets", fake_secrets)
    result = resolve_anthropic_api_key(include_secrets=True, include_env=True)
    assert result == "secret-fallback"


def test_resolve_anthropic_key_secret_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Secrets path: CLAUDE_API_STRANSKE secret wins even when ANTHROPIC_API_KEY env is set."""
    import streamlit as st

    fake_secrets = {"CLAUDE_API_STRANSKE": "secret-primary"}
    monkeypatch.setattr(st, "secrets", fake_secrets)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-fallback")
    result = resolve_anthropic_api_key(include_secrets=True, include_env=True)
    assert result == "secret-primary"


def test_anthropic_status_secrets_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """anthropic_api_key_status reflects secrets presence."""
    import streamlit as st

    fake_secrets = {"CLAUDE_API_STRANSKE": "secret-val"}
    monkeypatch.setattr(st, "secrets", fake_secrets)
    status = anthropic_api_key_status(include_secrets=True, include_env=True)
    assert status["CLAUDE_API_STRANSKE"] is True


def test_config_error_message_mentions_precedence() -> None:
    """Error message on missing key mentions CLAUDE_API_STRANSKE as preferred."""
    with pytest.raises(ValueError, match=r"CLAUDE_API_STRANSKE \(preferred\)"):
        resolve_llm_provider_config("anthropic")
