from __future__ import annotations

import os

import pytest

from trend_analysis.llm import tracing as tracing_module
from trend_analysis.llm.tracing import (
    append_fleet_record,
    langsmith_tracing_context,
    load_fleet_records,
    maybe_enable_langsmith_tracing,
    resolve_trace_url,
    stable_hash,
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
    monkeypatch.setenv("TREND_LANGSMITH_TRACE_TESTS", "1")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    tracing_module._LANGSMITH_ENABLED = None

    calls: dict[str, object] = {}

    class DummyRun:
        def __init__(self) -> None:
            self.outputs: dict[str, str] | None = None

        def end(
            self, *, outputs: dict[str, str] | None = None, error: str | None = None
        ) -> None:
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

    run_helpers = pytest.importorskip("langsmith.run_helpers")

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


def test_resolve_trace_url_prefers_property() -> None:
    class DummyRun:
        url = "https://example.test/run/123"

    assert resolve_trace_url(DummyRun()) == "https://example.test/run/123"


def test_resolve_trace_url_falls_back_to_method() -> None:
    class DummyRun:
        def get_url(self) -> str:
            return "https://example.test/run/456"

    assert resolve_trace_url(DummyRun()) == "https://example.test/run/456"


def test_stable_hash_normalizes_sets_deterministically() -> None:
    left = {"values": {"beta", "alpha"}, "nested": [{"ids": {3, 1, 2}}]}
    right = {"nested": [{"ids": {2, 3, 1}}], "values": {"alpha", "beta"}}

    assert stable_hash(left) == stable_hash(right)


def test_load_fleet_records_filters_schema_and_invalid_lines(tmp_path) -> None:
    fleet_path = tmp_path / "fleet.ndjson"
    append_fleet_record(
        {
            "schema_version": "langsmith-fleet/v1",
            "operation": "nl_to_patch",
            "status": "success",
        },
        path=fleet_path,
    )
    with fleet_path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema_version":"langsmith-fleet/v0","operation":"old"}\n')
        handle.write("{not-json}\n")
        handle.write("\n")

    records = load_fleet_records(path=fleet_path)

    assert len(records) == 1
    assert records[0]["operation"] == "nl_to_patch"
