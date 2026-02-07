"""Tests for the NL operation log viewer component."""

from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from trend_analysis.config.patch import ConfigPatch, PatchOperation
from trend_analysis.llm.nl_logging import NLOperationLog


def _load_module(monkeypatch: pytest.MonkeyPatch):
    st_stub = MagicMock()
    st_stub.session_state = {}
    monkeypatch.setitem(sys.modules, "streamlit", st_stub)
    return importlib.reload(importlib.import_module("streamlit_app.components.nl_operation_viewer"))


def test_sanitize_prompt_variables_redacts_sensitive_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    payload = {
        "api_key": "sk-test-key-1234567890",
        "token": "super-secret",
        "nested": {"secret": "still-secret"},
        "safe": "hello",
    }

    sanitized = module._sanitize_prompt_variables(payload)

    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["token"] == "[REDACTED]"
    assert sanitized["nested"]["secret"] == "[REDACTED]"
    assert sanitized["safe"] == "hello"


def test_sanitize_prompt_variables_redacts_sensitive_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    payload = {
        "headers": [
            "Authorization: Bearer sk-test-1234567890abcdef1234567890",
            "x-api-key: my-key",
        ],
        "items": [{"token": "top-secret"}, "plain text"],
    }

    sanitized = module._sanitize_prompt_variables(payload)

    assert "[REDACTED]" in sanitized["headers"][0]
    assert "[REDACTED]" in sanitized["headers"][1]
    assert sanitized["items"][0]["token"] == "[REDACTED]"
    assert sanitized["items"][1] == "plain text"


def test_redact_url_strips_query(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    url = "https://example.com/path?token=secret#frag"

    redacted = module._redact_url(url)

    assert redacted == "https://example.com/path"


def _make_entry(**kwargs) -> NLOperationLog:
    defaults = dict(
        request_id="req-1",
        timestamp=datetime.now(timezone.utc),
        operation="nl_to_patch",
        input_hash="hash-1",
        prompt_template="Prompt",
        prompt_variables={},
        model_output="output",
        parsed_patch=None,
        validation_result=None,
        error=None,
        duration_ms=12.5,
        model_name="gpt-4o-mini",
        temperature=0.2,
        token_usage=None,
        trace_url=None,
    )
    defaults.update(kwargs)
    return NLOperationLog(**defaults)


def test_prepare_replay_entry_redacts_sensitive_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    entry = _make_entry(
        prompt_template="Authorization: Bearer sk-test-1234567890abcdef1234567890",
        prompt_variables={"api_key": "sk-test-1234567890abcdef1234567890"},
    )

    redacted_entry, redacted = module._prepare_replay_entry(entry)

    assert redacted is True
    assert "[REDACTED]" in (redacted_entry.prompt_template or "")
    assert redacted_entry.prompt_variables["api_key"] == "[REDACTED]"


def test_prepare_replay_entry_no_redaction_when_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    entry = _make_entry(prompt_template="Hello", prompt_variables={"user": "test"})

    redacted_entry, redacted = module._prepare_replay_entry(entry)

    assert redacted is False
    assert redacted_entry.prompt_template == "Hello"
    assert redacted_entry.prompt_variables == {"user": "test"}


def test_render_prompt_for_display_redacts_by_key(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    entry = _make_entry(
        prompt_template="Password: {password}",
        prompt_variables={"password": "hunter2"},
    )

    rendered = module._render_prompt_for_display(entry)

    assert "hunter2" not in rendered
    assert "[REDACTED]" in rendered


def test_sanitize_patch_payload_redacts_sensitive_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    patch = ConfigPatch(
        summary="Set api key to sk-test-secret",
        operations=[
            PatchOperation(op="set", path="credentials.api_key", value="hunter2"),
        ],
    )

    payload = module._sanitize_patch_payload(patch)

    assert "[REDACTED]" in payload["summary"]
    assert payload["operations"][0]["value"] == "[REDACTED]"


def test_build_diff_summary_formats_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    patch = ConfigPatch(
        summary="Adjust volatility floor",
        operations=[
            PatchOperation(op="set", path="risk.vol_floor", value=0.15),
            PatchOperation(op="set", path="risk.warmup_periods", value=5),
        ],
    )

    payload = module._sanitize_patch_payload(patch)
    summary = module._build_diff_summary(payload)

    assert summary == [
        'set risk.vol_floor -> 0.15',
        'set risk.warmup_periods -> 5',
    ]


def test_load_log_entries_respects_limit(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    log_path = tmp_path / "nl_ops_2026-02-03.jsonl"

    for index in range(3):
        entry = _make_entry(
            request_id=f"req-{index}",
            input_hash=f"hash-{index}",
        )
        line = json.dumps(entry.model_dump(mode="json"), separators=(",", ":"))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    loaded = module._load_log_entries(log_path, limit=2)

    assert [entry_index for entry_index, _ in loaded] == [2, 3]
