"""LangSmith tracing helpers for NL operations."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

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


@contextmanager
def langsmith_tracing_context() -> Iterator[None]:
    """Provide a tracing context when LangSmith is enabled."""

    if not maybe_enable_langsmith_tracing():
        yield
        return
    try:
        from langchain_core.tracers.context import tracing_v2_enabled
    except Exception:
        yield
        return
    project = os.environ.get("LANGCHAIN_PROJECT") or os.environ.get("LANGSMITH_PROJECT")
    if project:
        try:
            with tracing_v2_enabled(project_name=project):
                yield
        except TypeError:
            with tracing_v2_enabled():
                yield
    else:
        with tracing_v2_enabled():
            yield


__all__ = ["langsmith_tracing_context", "maybe_enable_langsmith_tracing"]
