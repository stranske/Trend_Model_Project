"""Tests for the NL operation log viewer component."""

from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

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


def test_load_log_entries_respects_limit(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    log_path = tmp_path / "nl_ops_2026-02-03.jsonl"

    for index in range(3):
        entry = NLOperationLog(
            request_id=f"req-{index}",
            timestamp=datetime.now(timezone.utc),
            operation="nl_to_patch",
            input_hash=f"hash-{index}",
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
        line = json.dumps(entry.model_dump(mode="json"), separators=(",", ":"))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    loaded = module._load_log_entries(log_path, limit=2)

    assert [entry_index for entry_index, _ in loaded] == [2, 3]
