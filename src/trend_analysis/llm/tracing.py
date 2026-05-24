"""LangSmith tracing helpers for NL operations."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Literal

_LANGSMITH_ENABLED: bool | None = None
_TRUTHY = {"1", "true", "yes", "on"}
FLEET_SCHEMA_VERSION = "langsmith-fleet/v1"
FLEET_SOURCE_REPO = "stranske/Trend_Model_Project"
DEFAULT_FLEET_ARTIFACT_PATH = "artifacts/langsmith/langsmith-fleet.ndjson"


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


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


def stable_hash(value: Any, *, prefix: str = "sha256:") -> str:
    """Return a deterministic digest for metadata without storing raw content."""

    try:
        serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        serialized = str(value)
    return f"{prefix}{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"


def default_fleet_artifact_path() -> Path:
    return Path(os.environ.get("TREND_LANGSMITH_FLEET_PATH", DEFAULT_FLEET_ARTIFACT_PATH))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def build_fleet_record(
    *,
    operation: str,
    status: str,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    trace_url: str | None = None,
    latency_ms: float | None = None,
    error_category: str | None = None,
    domain: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a dashboard-compatible LangSmith fleet record."""

    return {
        "schema_version": FLEET_SCHEMA_VERSION,
        "source_repo": FLEET_SOURCE_REPO,
        "timestamp": datetime.now(UTC).isoformat(),
        "operation": operation,
        "status": status,
        "provider": provider or "unknown",
        "model": model or "unknown",
        "temperature": temperature,
        "trace_url": trace_url,
        "latency_ms": latency_ms,
        "error_category": error_category,
        "domain": _json_safe(domain or {}),
    }


def append_fleet_record(record: dict[str, Any], *, path: str | Path | None = None) -> None:
    output_path = Path(path) if path is not None else default_fleet_artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_json_safe(record), sort_keys=True, separators=(",", ":"))
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def record_fleet_event(**kwargs: Any) -> dict[str, Any]:
    record = build_fleet_record(**kwargs)
    try:
        append_fleet_record(record)
    except Exception:
        pass
    return record


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

    if os.environ.get("PYTEST_CURRENT_TEST") and not _truthy_env("TREND_LANGSMITH_TRACE_TESTS"):
        yield None
        return
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


__all__ = [
    "FLEET_SCHEMA_VERSION",
    "append_fleet_record",
    "build_fleet_record",
    "default_fleet_artifact_path",
    "langsmith_tracing_context",
    "maybe_enable_langsmith_tracing",
    "record_fleet_event",
    "resolve_trace_url",
    "stable_hash",
]
