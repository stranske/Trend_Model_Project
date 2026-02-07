"""Tests for the NL operation log viewer component."""

from __future__ import annotations

import importlib
import json
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from trend_analysis.config.patch import ConfigPatch, PatchOperation
from trend_analysis.llm.nl_logging import NLOperationLog


def _load_module(monkeypatch: pytest.MonkeyPatch):
    st_stub = MagicMock()
    st_stub.session_state = {}
    st_stub.expander.side_effect = lambda *_, **__: nullcontext()
    st_stub.spinner.side_effect = lambda *_, **__: nullcontext()
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


def test_prepare_replay_entry_redacts_sensitive_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    entry = _make_entry(
        prompt_template="Authorization: Bearer sk-test-1234567890abcdef1234567890",
        prompt_variables={"api_key": "sk-test-1234567890abcdef1234567890"},
    )

    redacted_entry, redacted = module._prepare_replay_entry(entry)

    assert redacted is True
    assert "[REDACTED]" in (redacted_entry.prompt_template or "")
    assert redacted_entry.prompt_variables["api_key"] == "[REDACTED]"


def test_prepare_replay_entry_no_redaction_when_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    entry = _make_entry(prompt_template="Hello", prompt_variables={"user": "test"})

    redacted_entry, redacted = module._prepare_replay_entry(entry)

    assert redacted is False
    assert redacted_entry.prompt_template == "Hello"
    assert redacted_entry.prompt_variables == {"user": "test"}


def test_render_prompt_for_display_redacts_by_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        "set risk.vol_floor -> 0.15",
        "set risk.warmup_periods -> 5",
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


def test_load_log_entries_orders_by_timestamp(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    log_path = tmp_path / "nl_ops_2026-02-03.jsonl"
    t1 = datetime(2026, 2, 3, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 2, 3, 12, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 2, 3, 14, 0, tzinfo=timezone.utc)

    entries = [
        _make_entry(request_id="req-2", input_hash="hash-2", timestamp=t2),
        _make_entry(request_id="req-1", input_hash="hash-1", timestamp=t1),
        _make_entry(request_id="req-3", input_hash="hash-3", timestamp=t3),
    ]
    for entry in entries:
        line = json.dumps(entry.model_dump(mode="json"), separators=(",", ":"))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    loaded = module._load_log_entries(log_path, limit=50)

    assert [entry.request_id for _, entry in loaded] == ["req-1", "req-2", "req-3"]


def test_load_log_entries_returns_most_recent_50(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    log_path = tmp_path / "nl_ops_2026-02-03.jsonl"

    base_time = datetime(2026, 2, 3, 10, 0, tzinfo=timezone.utc)
    for offset in range(55):
        entry = _make_entry(
            request_id=f"req-{offset}",
            input_hash=f"hash-{offset}",
            timestamp=base_time.replace(minute=offset),
        )
        line = json.dumps(entry.model_dump(mode="json"), separators=(",", ":"))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    loaded = module._load_log_entries(log_path, limit=50)

    assert len(loaded) == 50
    assert loaded[0][0] == 6
    assert loaded[-1][0] == 55


def test_load_log_entries_returns_all_when_under_limit(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module(monkeypatch)
    log_path = tmp_path / "nl_ops_2026-02-03.jsonl"

    base_time = datetime(2026, 2, 3, 9, 0, tzinfo=timezone.utc)
    for offset in range(10):
        entry = _make_entry(
            request_id=f"req-{offset}",
            input_hash=f"hash-{offset}",
            timestamp=base_time.replace(minute=offset),
        )
        line = json.dumps(entry.model_dump(mode="json"), separators=(",", ":"))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    loaded = module._load_log_entries(log_path, limit=50)

    assert len(loaded) == 10


def test_redact_text_replaces_fixtures_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    fixture_text = (
        "sk-proj-abc123xyz ghp_abc123xyz github_pat_abc123xyz "
        "AKIAIOSFODNN7EXAMPLE SECRET_KEY=value123"
    )

    redacted = module._redact_text(fixture_text)

    assert "sk-proj-abc123xyz" not in redacted
    assert "ghp_abc123xyz" not in redacted
    assert "github_pat_abc123xyz" not in redacted
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted
    assert "SECRET_KEY=value123" not in redacted
    assert "[REDACTED]" in redacted
    assert module._redact_text(redacted) == redacted


def test_render_replay_stores_result_and_renders_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    st_stub = sys.modules["streamlit"]
    entry = _make_entry()
    replay_result = MagicMock(
        output="result sk-proj-abc123xyz",
        diff="diff ghp_abc123xyz",
        trace_url="https://example.com/path?token=secret",
    )
    replay_mock = MagicMock(return_value=replay_result)
    monkeypatch.setattr(module, "replay_nl_entry", replay_mock)
    st_stub.selectbox.return_value = "openai"
    st_stub.text_input.return_value = entry.model_name or ""
    st_stub.slider.return_value = float(entry.temperature or 0.0)

    module._render_replay(entry, entry_id="1", run_replay=True)

    replay_mock.assert_called_once()
    assert st_stub.session_state["nl_replay_result_1"]["output"] == replay_result.output
    assert st_stub.expander.call_args_list[-1].args[0] == "Replay Results"
    code_calls = [call.args[0] for call in st_stub.code.call_args_list]
    assert any("[REDACTED]" in text for text in code_calls)


def test_replay_button_invokes_replay_for_selected_entry(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module(monkeypatch)
    st_stub = sys.modules["streamlit"]

    log_dir = tmp_path / ".trend_nl_logs"
    log_dir.mkdir()
    log_path = log_dir / "nl_ops_2026-02-03.jsonl"
    entry = _make_entry()
    line = json.dumps(entry.model_dump(mode="json"), separators=(",", ":"))
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(line + "\n")

    label = (
        f"1. {module._format_timestamp(entry)} | {entry.operation} | "
        f"{entry.model_name or 'unknown'} | {module._format_duration(entry)}"
    )
    st_stub.selectbox.side_effect = [log_path.name, label, "openai"]
    st_stub.text_input.return_value = entry.model_name or ""
    st_stub.slider.return_value = float(entry.temperature or 0.0)
    st_stub.button.return_value = True

    replay_result = MagicMock(output="ok", diff=None, trace_url=None)
    replay_mock = MagicMock(return_value=replay_result)
    monkeypatch.setattr(module, "replay_nl_entry", replay_mock)

    module.render_nl_operation_viewer(base_dir=log_dir)

    replay_mock.assert_called_once()
    called_entry = replay_mock.call_args.args[0]
    entry_id = f"{log_path.stem}_1"
    assert called_entry.request_id == entry.request_id
    assert st_stub.session_state[f"nl_replay_result_{entry_id}"]["output"] == "ok"


def test_replay_session_state_keys_are_entry_specific(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module(monkeypatch)
    st_stub = sys.modules["streamlit"]

    log_dir = tmp_path / ".trend_nl_logs"
    log_dir.mkdir()
    log_path = log_dir / "nl_ops_2026-02-03.jsonl"
    entry = _make_entry()
    line = json.dumps(entry.model_dump(mode="json"), separators=(",", ":"))
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(line + "\n")

    label = (
        f"1. {module._format_timestamp(entry)} | {entry.operation} | "
        f"{entry.model_name or 'unknown'} | {module._format_duration(entry)}"
    )
    st_stub.selectbox.side_effect = [log_path.name, label, "openai"]
    st_stub.text_input.return_value = entry.model_name or ""
    st_stub.slider.return_value = float(entry.temperature or 0.0)
    st_stub.button.return_value = True

    replay_result = MagicMock(output="ok", diff=None, trace_url=None)
    replay_mock = MagicMock(return_value=replay_result)
    monkeypatch.setattr(module, "replay_nl_entry", replay_mock)

    module.render_nl_operation_viewer(base_dir=log_dir)

    assert "nl_replay_open" not in st_stub.session_state
    assert "nl_replay_result" not in st_stub.session_state
