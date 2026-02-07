"""LangSmith tracing helpers for NL operations."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, Literal

_LANGSMITH_ENABLED: bool | None = None


def maybe_enable_langsmith_tracing() -> bool:
    """Enable LangSmith tracing when LANGSMITH_API_KEY is present."""

    global _LANGSMITH_ENABLED
    if _LANGSMITH_ENABLED is not None:
        return _LANGSMITH_ENABLED
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        _LANGSMITH_ENABLED = False
        return False
    if not os.environ.get("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    _LANGSMITH_ENABLED = True
    return True


def _get_langsmith_project() -> str | None:
    """Return the configured LangSmith project name, if any."""
    return os.environ.get("LANGCHAIN_PROJECT") or os.environ.get("LANGSMITH_PROJECT")


def resolve_trace_url(run: Any) -> str | None:
    """Resolve the trace URL from a LangSmith run object."""

    if run is None:
        return None
    url_attr = getattr(run, "url", None)
    if isinstance(url_attr, str) and url_attr:
        return url_attr
    if callable(url_attr):
        try:
            value = url_attr()
        except TypeError:
            value = None
        if isinstance(value, str) and value:
            return value
    for method_name in ("get_url", "get_run_url"):
        method = getattr(run, method_name, None)
        if not callable(method):
            continue
        try:
            value = method()
        except TypeError:
            value = None
        if isinstance(value, str) and value:
            return value
    return None


@contextmanager
def langsmith_tracing_context(
    *,
    name: str = "nl_operation",
    run_type: Literal[
        "retriever", "llm", "tool", "chain", "embedding", "prompt", "parser"
    ] = "chain",
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Provide a LangSmith tracing context and optional run metadata."""

    if not maybe_enable_langsmith_tracing():
        yield None
        return
    try:
        from langsmith import run_helpers
    except Exception:
        yield None
        return
    project: str | None = _get_langsmith_project()
    try:
        trace_cm = run_helpers.trace(
            name,
            run_type=run_type,
            inputs=inputs,
            metadata=metadata,
            project_name=project,
        )
    except Exception:
        yield None
        return
    try:
        from langchain_core.tracers.context import tracing_v2_enabled
    except Exception:
        with trace_cm as run:
            yield run
        return
    with trace_cm as run:
        if project:
            try:
                with tracing_v2_enabled(project_name=project):
                    yield run
            except TypeError:
                with tracing_v2_enabled():
                    yield run
        else:
            with tracing_v2_enabled():
                yield run


__all__ = ["langsmith_tracing_context", "maybe_enable_langsmith_tracing", "resolve_trace_url"]
