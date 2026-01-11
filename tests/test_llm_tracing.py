from __future__ import annotations

import os

from trend_analysis.llm import tracing as tracing_module
from trend_analysis.llm.tracing import langsmith_tracing_context, maybe_enable_langsmith_tracing


def test_langsmith_tracing_disabled_without_key(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    tracing_module._LANGSMITH_ENABLED = None

    enabled = maybe_enable_langsmith_tracing()

    assert enabled is False
    assert os.environ.get("LANGCHAIN_API_KEY") is None
    assert os.environ.get("LANGCHAIN_TRACING_V2") is None


def test_langsmith_tracing_enabled_with_key(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    tracing_module._LANGSMITH_ENABLED = None

    enabled = maybe_enable_langsmith_tracing()

    assert enabled is True
    assert os.environ.get("LANGCHAIN_API_KEY") == "test-key"
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"


def test_langsmith_tracing_context_is_noop_without_key(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    tracing_module._LANGSMITH_ENABLED = None

    with langsmith_tracing_context():
        pass

    assert os.environ.get("LANGCHAIN_TRACING_V2") is None
