"""Streamlit viewer for NL operation logs and replay."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

import streamlit as st

from trend_analysis.llm.nl_logging import NLOperationLog
from trend_analysis.llm.replay import render_prompt, replay_nl_entry
from trend_analysis.logging import iter_jsonl

_LOG_FILE_GLOB = "nl_ops_*.jsonl"
_DEFAULT_MAX_ENTRIES = 200
_REDACT_KEYS = (
    "key",
    "token",
    "secret",
    "password",
    "credential",
    "api_key",
    "apikey",
    "access_key",
    "authorization",
    "bearer",
    "session",
    "cookie",
)
_REDACT_TEXT_PATTERNS = [
    re.compile(
        r"-----BEGIN [A-Z ]+ PRIVATE KEY-----.*?-----END [A-Z ]+ PRIVATE KEY-----",
        re.DOTALL,
    ),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"sk-ant-[A-Za-z0-9-]{20,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*"),
    re.compile(r"(?i)(api[-_]?key|token|secret|password|authorization)\s*[:=]\s*\S+"),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    re.compile(r"(AKIA|ASIA)[0-9A-Z]{16}"),
]


@dataclass(frozen=True)
class _LogChoice:
    label: str
    entry: NLOperationLog
    index: int


def _list_log_files(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    return sorted(base_dir.glob(_LOG_FILE_GLOB), reverse=True)


def _load_log_entries(
    path: Path, limit: int = _DEFAULT_MAX_ENTRIES
) -> list[tuple[int, NLOperationLog]]:
    entries: list[tuple[int, NLOperationLog]] = []
    for index, payload in enumerate(iter_jsonl(path), start=1):
        try:
            entry = NLOperationLog.model_validate(payload)
        except Exception:
            continue
        entries.append((index, entry))
    if limit and len(entries) > limit:
        entries = entries[-limit:]
    return entries


def _redact_text(text: str | None) -> str:
    if not text:
        return ""
    redacted = text
    for pattern in _REDACT_TEXT_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in _REDACT_KEYS)


def _sanitize_value(value: Any, *, key: str | None = None) -> Any:
    if key and _is_sensitive_key(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _sanitize_value(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _sanitize_prompt_variables(payload: dict[str, Any]) -> dict[str, Any]:
    return _sanitize_value(payload)


def _contains_sensitive(value: Any, *, key: str | None = None) -> bool:
    if key and _is_sensitive_key(key):
        return True
    if isinstance(value, dict):
        return any(_contains_sensitive(v, key=str(k)) for k, v in value.items())
    if isinstance(value, list):
        return any(_contains_sensitive(item) for item in value)
    if isinstance(value, str):
        return _redact_text(value) != value
    return False


def _redact_url(url: str | None) -> str | None:
    if not url:
        return None
    redacted = _redact_text(url)
    try:
        parts = urlsplit(redacted)
    except ValueError:
        return redacted
    if not parts.scheme or not parts.netloc:
        return redacted
    netloc = parts.netloc.split("@", 1)[-1]
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _redact_entry_for_replay(entry: NLOperationLog) -> NLOperationLog:
    return entry.model_copy(
        update={
            "prompt_template": _redact_text(entry.prompt_template or ""),
            "prompt_variables": _sanitize_prompt_variables(entry.prompt_variables or {}),
        }
    )


def _entry_has_sensitive_prompt(entry: NLOperationLog) -> bool:
    if _redact_text(entry.prompt_template or "") != (entry.prompt_template or ""):
        return True
    return _contains_sensitive(entry.prompt_variables or {})



def _format_timestamp(entry: NLOperationLog) -> str:
    timestamp = entry.timestamp
    try:
        return timestamp.isoformat(timespec="seconds")
    except Exception:
        return str(timestamp)


def _format_duration(entry: NLOperationLog) -> str:
    try:
        return f"{entry.duration_ms:.0f}ms"
    except Exception:
        return "—"


def _build_choices(entries: Iterable[tuple[int, NLOperationLog]]) -> list[_LogChoice]:
    choices: list[_LogChoice] = []
    for index, entry in entries:
        label = (
            f"{index}. {_format_timestamp(entry)} | {entry.operation} | "
            f"{entry.model_name or 'unknown'} | {_format_duration(entry)}"
        )
        choices.append(_LogChoice(label=label, entry=entry, index=index))
    return choices


def _render_entry_table(choices: list[_LogChoice]) -> None:
    rows: list[dict[str, str]] = []
    for choice in choices:
        entry = choice.entry
        rows.append(
            {
                "Entry": str(choice.index),
                "Timestamp": _format_timestamp(entry),
                "Operation": str(entry.operation),
                "Model": str(entry.model_name or "unknown"),
                "Duration": _format_duration(entry),
                "Trace URL": str(_redact_url(entry.trace_url) or "—"),
            }
        )
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_patch_summary(entry: NLOperationLog) -> None:
    patch = entry.parsed_patch
    if patch is None:
        st.info("No parsed patch recorded for this entry.")
        return
    st.markdown("**Patch summary**")
    st.caption(patch.summary)
    if patch.risk_flags:
        st.caption("Risk flags: " + ", ".join(flag.value for flag in patch.risk_flags))
    st.caption("Needs review: " + ("Yes" if patch.needs_review else "No"))
    operations = [
        f"{op.op} {op.path} -> {json.dumps(op.value, default=str)}" for op in patch.operations
    ]
    if operations:
        st.code("\n".join(operations), language="text")
    st.markdown("**Patch payload**")
    st.code(json.dumps(patch.model_dump(), indent=2, sort_keys=True), language="json")


def _render_replay(entry: NLOperationLog) -> None:
    st.markdown("**Replay**")
    st.caption("Replays may differ across time, models, or provider settings.")
    has_sensitive = _entry_has_sensitive_prompt(entry)
    if has_sensitive:
        st.warning(
            "Sensitive data detected in the prompt. Replay will use a redacted prompt to prevent leakage."
        )
        redact_prompt = True
    else:
        redact_prompt = st.checkbox(
            "Redact sensitive data in replay prompt (recommended)",
            value=True,
            key="nl_replay_redact_prompt",
        )
    provider = st.selectbox("Provider", ["openai", "anthropic", "ollama"], key="nl_replay_provider")
    model = st.text_input("Model (optional)", value=entry.model_name or "", key="nl_replay_model")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=float(entry.temperature or 0.0),
        step=0.05,
        key="nl_replay_temperature",
    )
    clicked = st.button("Replay entry (outputs may differ)", key="nl_replay_btn")
    if not clicked:
        return
    with st.spinner("Replaying entry..."):
        try:
            replay_entry = _redact_entry_for_replay(entry) if redact_prompt else entry
            replay_result = replay_nl_entry(
                replay_entry,
                provider=provider,
                model=model or None,
                temperature=temperature,
            )
        except Exception as exc:
            st.error("Replay failed. Ensure provider credentials are available.")
            st.caption(str(exc))
            return
    st.success("Replay completed.")
    st.markdown("**Replay output**")
    st.code(_redact_text(replay_result.output), language="text")
    if replay_result.trace_url:
        st.caption(f"Trace URL: {_redact_url(replay_result.trace_url)}")
    if replay_result.diff:
        st.markdown("**Diff vs recorded output**")
        st.code(_redact_text(replay_result.diff), language="diff")


def render_nl_operation_viewer(
    *,
    base_dir: Path | None = None,
    max_entries: int = _DEFAULT_MAX_ENTRIES,
) -> None:
    """Render a viewer for recent NL operation logs."""

    log_dir = base_dir or Path(".trend_nl_logs")
    if not log_dir.exists():
        st.info("No NL log directory found yet.")
        return

    log_files = _list_log_files(log_dir)
    if not log_files:
        st.info("No NL operation logs found.")
        return

    st.markdown("**Recent NL operations**")
    file_labels = [path.name for path in log_files]
    selected_label = st.selectbox("Log file", file_labels, key="nl_log_file_select")
    selected_path = log_files[file_labels.index(selected_label)]
    entries = _load_log_entries(selected_path, limit=max_entries)
    if not entries:
        st.info("Selected log file has no readable entries.")
        return

    ordered = list(reversed(entries))
    choices = _build_choices(ordered)
    _render_entry_table(choices)

    labels = [choice.label for choice in choices]
    selected_entry_label = st.selectbox("Select entry", labels, key="nl_log_entry_select")
    selected_choice = next(choice for choice in choices if choice.label == selected_entry_label)
    entry = selected_choice.entry

    st.markdown("**Entry details**")
    st.caption(f"Request ID: {entry.request_id}")
    if entry.error:
        st.warning(f"Error: {entry.error}")
    st.caption(
        f"Operation: {entry.operation} | Model: {entry.model_name} | Duration: {_format_duration(entry)}"
    )
    if entry.trace_url:
        st.caption(f"Trace URL: {_redact_url(entry.trace_url)}")

    prompt = _redact_text(render_prompt(entry))
    st.markdown("**Rendered prompt**")
    st.code(prompt, language="text")

    variables = entry.prompt_variables or {}
    if variables:
        st.markdown("**Prompt variables**")
        st.code(json.dumps(_sanitize_prompt_variables(dict(variables)), indent=2), language="json")

    if entry.model_output:
        st.markdown("**Model output**")
        st.code(_redact_text(entry.model_output), language="text")

    _render_patch_summary(entry)
    with st.expander("Replay entry", expanded=False):
        _render_replay(entry)


__all__ = [
    "render_nl_operation_viewer",
]
