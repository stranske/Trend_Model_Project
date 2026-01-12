from __future__ import annotations

import os

from trend_analysis.llm import tracing as tracing_module
from trend_analysis.llm.tracing import (
    langsmith_tracing_context,
    maybe_enable_langsmith_tracing,
)


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


def test_langsmith_tracing_context_invokes_trace(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    tracing_module._LANGSMITH_ENABLED = None

    calls: dict[str, object] = {}

    class DummyRun:
        def __init__(self) -> None:
            self.outputs: dict[str, str] | None = None

        def end(self, *, outputs: dict[str, str] | None = None, error: str | None = None) -> None:
            self.outputs = outputs
            assert error is None

    class DummyTrace:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls["args"] = args
            calls["kwargs"] = kwargs
            self._run = DummyRun()

        def __enter__(self) -> DummyRun:
            return self._run

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> bool:
            return False

    import langsmith.run_helpers as run_helpers

    monkeypatch.setattr(run_helpers, "trace", DummyTrace)

    with langsmith_tracing_context(
        name="nl_to_patch",
        inputs={"prompt": "hello"},
        metadata={"request_id": "req-123"},
    ) as run:
        assert run is not None
        run.end(outputs={"output": "ok"})

    assert calls["args"] == ("nl_to_patch",)
    assert isinstance(calls["kwargs"], dict)
    assert calls["kwargs"]["inputs"] == {"prompt": "hello"}
    assert calls["kwargs"]["metadata"] == {"request_id": "req-123"}
