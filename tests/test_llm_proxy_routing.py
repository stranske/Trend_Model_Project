"""Tests for internal/on-prem LLM routing: proxy base_url, zone guard, egress.

Covers issue #5344 (internal browser deployment that keeps proprietary data and
the LLM inside the org perimeter):

* ``test_app_uses_base_url_from_env`` locks the contract that the Streamlit LLM
  settings honor ``TREND_LLM_BASE_URL`` so the app talks to the in-perimeter
  proxy instead of the public API.
* ``test_zone_disabled_hides_llm`` proves the guarded LLM components construct
  no provider and render no panels when ``TREND_LLM_ZONE=disabled``.
* ``test_proxy_request_body_within_limit`` proves the proxy's data-egress
  ceiling (``TS_LLM_PROXY_MAX_BODY_BYTES``) bounds what leaves the perimeter.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

_ENV_KEYS = (
    "TREND_LLM_BASE_URL",
    "TREND_LLM_ZONE",
    "TREND_LLM_PROVIDER",
    "TREND_LLM_API_KEY",
    "TS_STREAMLIT_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "CLAUDE_API_STRANSKE",
    "TS_LLM_PROXY_MAX_BODY_BYTES",
    "TS_LLM_PROXY_TOKEN",
    "TS_LLM_PROXY_URL",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_app_uses_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``TREND_LLM_BASE_URL`` set, the resolved config carries that base_url."""

    from streamlit_app.components.llm_settings import resolve_llm_provider_config

    monkeypatch.setenv("TREND_LLM_BASE_URL", "http://llm-proxy:8799/v1")
    config = resolve_llm_provider_config("openai", api_key="sk-test")

    assert config.base_url == "http://llm-proxy:8799/v1"


def test_proxy_token_used_for_internal_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Internal proxy routing uses the proxy token as the client API key."""

    from streamlit_app.components.llm_settings import (
        default_api_key,
        resolve_llm_provider_config,
    )

    monkeypatch.setenv("TREND_LLM_BASE_URL", "http://llm-proxy:8799/v1")
    monkeypatch.setenv("TS_LLM_PROXY_TOKEN", "proxy-token")

    assert default_api_key("openai") == "proxy-token"
    config = resolve_llm_provider_config("openai")
    assert config.base_url == "http://llm-proxy:8799/v1"
    assert config.api_key == "proxy-token"


def _reload_with_stub(module_name: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    st_stub = MagicMock()
    st_stub.session_state = {}
    monkeypatch.setitem(sys.modules, "streamlit", st_stub)
    module = importlib.reload(importlib.import_module(module_name))
    return module, st_stub


@dataclass
class _StubResult:
    details: dict[str, Any]


def test_zone_disabled_hides_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """TREND_LLM_ZONE=disabled hides LLM panels and builds no provider."""

    monkeypatch.setenv("TREND_LLM_ZONE", "disabled")
    details = {"out_sample_stats": {"Portfolio": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)}}

    # explain_results component
    explain, explain_st = _reload_with_stub(
        "streamlit_app.components.explain_results", monkeypatch
    )
    provider_calls: list[Any] = []
    monkeypatch.setattr(
        explain, "_resolve_llm_provider_config", lambda *a, **k: provider_calls.append((a, k))
    )
    explain.render_explain_results(_StubResult(details=details), run_key="rk")
    assert explain_st.info.called, "disabled zone should surface an info notice"
    assert not explain_st.text_area.called, "no LLM panels should render when disabled"
    assert provider_calls == [], "no LLM provider should be constructed when disabled"

    # comparison_llm component
    comparison, comparison_st = _reload_with_stub(
        "streamlit_app.components.comparison_llm", monkeypatch
    )
    comparison_calls: list[Any] = []
    monkeypatch.setattr(
        comparison, "resolve_llm_provider_config", lambda *a, **k: comparison_calls.append((a, k))
    )
    comparison.render_comparison_llm(
        result_a=_StubResult(details=details),
        result_b=_StubResult(details=details),
        label_a="A",
        label_b="B",
        config_diff="",
        run_key="rk",
    )
    assert comparison_st.info.called
    assert not comparison_st.text_area.called
    assert comparison_calls == []


def test_proxy_request_body_within_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The proxy egress ceiling bounds the payload that leaves the perimeter."""

    server = importlib.import_module("trend_analysis.llm_proxy.server")

    # Unset -> no limit (default behavior preserved).
    assert server.request_body_within_limit(b"x" * 100_000) is True

    monkeypatch.setenv("TS_LLM_PROXY_MAX_BODY_BYTES", "100")
    assert server.request_body_within_limit(b"x" * 100) is True
    assert server.request_body_within_limit(b"x" * 101) is False

    # Non-numeric / non-positive -> treated as no limit.
    monkeypatch.setenv("TS_LLM_PROXY_MAX_BODY_BYTES", "not-a-number")
    assert server.request_body_within_limit(b"x" * 100_000) is True
    monkeypatch.setenv("TS_LLM_PROXY_MAX_BODY_BYTES", "0")
    assert server.request_body_within_limit(b"x" * 100_000) is True
