from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

from trend_analysis.config.patch import ConfigPatch, PatchOperation
from trend_analysis.config.validation import ValidationResult
from trend_analysis.llm.nl_logging import NLOperationLog
from trend_analysis.llm.replay import load_nl_log_entry, render_prompt, replay_nl_entry


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.bound: dict[str, object] | None = None
        self.last_prompt: object | None = None

    def bind(self, **kwargs: object) -> "FakeLLM":
        self.bound = kwargs
        return self

    def invoke(self, prompt: object) -> SimpleNamespace:
        self.last_prompt = prompt
        return SimpleNamespace(content=self.response)


def _make_entry(request_id: str, *, output: str | None = "ok") -> NLOperationLog:
    patch = ConfigPatch(
        operations=[PatchOperation(op="set", path="analysis.top_n", value=12)],
        summary="Update top_n selection",
    )
    validation = ValidationResult(valid=True, errors=[], warnings=[])
    return NLOperationLog(
        request_id=request_id,
        timestamp=datetime(2025, 1, 5, 12, 0, tzinfo=timezone.utc),
        operation="nl_to_patch",
        input_hash="hash-123",
        prompt_template="Hello {name}",
        prompt_variables={"name": "Trend"},
        model_output=output,
        parsed_patch=patch,
        validation_result=validation,
        error=None,
        duration_ms=12.5,
        model_name="unit-test-model",
        temperature=0.0,
        token_usage=None,
    )


def test_load_nl_log_entry_reads_selected_line(tmp_path) -> None:
    log_path = tmp_path / "nl.jsonl"
    entries = [_make_entry("req-1"), _make_entry("req-2")]
    with log_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.model_dump(mode="json")) + "\n")

    loaded = load_nl_log_entry(log_path, 2)

    assert loaded.request_id == "req-2"


def test_render_prompt_formats_template() -> None:
    entry = _make_entry("req-3")

    prompt = render_prompt(entry)

    assert prompt == "Hello Trend"


def test_replay_nl_entry_compares_outputs() -> None:
    entry = _make_entry("req-4", output="match")
    fake_llm = FakeLLM("match")

    result = replay_nl_entry(entry, llm=fake_llm)

    assert result.matches is True
    assert result.output_hash == result.recorded_hash
    assert result.diff is None


def test_replay_nl_entry_reports_diff_on_mismatch() -> None:
    entry = _make_entry("req-5", output="match")
    fake_llm = FakeLLM("different")

    result = replay_nl_entry(entry, llm=fake_llm)

    assert result.matches is False
    assert result.diff is not None
    assert result.diff.startswith("--- recorded")


def test_replay_nl_entry_logs_langsmith_trace_url(monkeypatch, caplog) -> None:
    entry = _make_entry("req-6", output="match")
    fake_llm = FakeLLM("match")

    class FakeRun:
        url = "https://example.test/trace/abc"

        def end(self, outputs: dict[str, str]) -> None:
            self.outputs = outputs

    @contextmanager
    def _fake_context(**_kwargs):
        yield FakeRun()

    monkeypatch.setattr(
        "trend_analysis.llm.tracing.langsmith_tracing_context",
        _fake_context,
    )

    with caplog.at_level(logging.INFO, logger="trend_analysis.llm.replay"):
        replay_nl_entry(entry, llm=fake_llm)

    assert "LangSmith trace: https://example.test/trace/abc" in caplog.text
