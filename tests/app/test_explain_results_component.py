"""Tests for the Streamlit explain results helpers."""

from __future__ import annotations

import hashlib
import importlib
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from trend_analysis.llm import ResultSummaryResponse


@dataclass
class _StubChain:
    response: ResultSummaryResponse
    last_payload: dict | None = None

    def run(self, **kwargs):
        self.last_payload = kwargs
        return self.response


@pytest.fixture()
def explain_module(monkeypatch: pytest.MonkeyPatch):
    st_stub = MagicMock()
    st_stub.session_state = {}
    monkeypatch.setitem(sys.modules, "streamlit", st_stub)
    module = importlib.reload(importlib.import_module("streamlit_app.components.explain_results"))
    return module


def test_render_analysis_output_includes_summary_and_sections(explain_module) -> None:
    details = {
        "out_sample_stats": {"Portfolio": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)},
        "extra": {"foo": 1},
    }

    output = explain_module._render_analysis_output(details)

    assert "Summary table" in output
    assert "Available sections: extra, out_sample_stats" in output


def test_generate_result_explanation_uses_chain_and_disclaimer(
    explain_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    details = {"out_sample_stats": {"Portfolio": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)}}
    response = ResultSummaryResponse(
        text="CAGR was 10% [from out_sample_stats].",
        trace_url="trace://example",
    )
    stub = _StubChain(response)

    monkeypatch.setattr(explain_module, "_build_result_chain", lambda provider=None: stub)

    explanation = explain_module.generate_result_explanation(details, questions="Summarize")

    assert explanation.trace_url == "trace://example"
    assert explanation.metric_count == 6
    assert explanation.claim_issues == []
    assert "This is analytical output, not financial advice." in explanation.text
    assert stub.last_payload is not None
    assert "analysis_output" in stub.last_payload
    assert "metric_catalog" in stub.last_payload
    assert "questions" in stub.last_payload


def test_generate_result_explanation_handles_missing_metrics(
    explain_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail_chain(*_args, **_kwargs):
        raise AssertionError("Chain should not be built without metrics.")

    monkeypatch.setattr(explain_module, "_build_result_chain", _fail_chain)

    explanation = explain_module.generate_result_explanation({}, questions=None)

    assert explanation.metric_count == 0
    assert "No metrics were detected in the analysis output." in explanation.text
    assert "This is analytical output, not financial advice." in explanation.text


def test_resolve_explanation_run_id_prefers_details(explain_module) -> None:
    run_key = "run:abc123"
    details = {"run_id": "run-001"}
    assert explain_module._resolve_explanation_run_id(details, run_key) == "run-001"

    details = {"metadata": {"run_id": "run-002"}}
    assert explain_module._resolve_explanation_run_id(details, run_key) == "run-002"

    details = {"metadata": {"reporting": {"run_id": "run-003"}}}
    assert explain_module._resolve_explanation_run_id(details, run_key) == "run-003"

    details = {}
    expected = hashlib.sha256(run_key.encode("utf-8")).hexdigest()[:12]
    assert explain_module._resolve_explanation_run_id(details, run_key) == expected


def test_resolve_llm_provider_config_requires_api_key(
    explain_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TREND_LLM_PROVIDER", "openai")
    monkeypatch.delenv("TREND_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Missing API key.*OPENAI_API_KEY"):
        explain_module._resolve_llm_provider_config()


def test_render_explain_results_uses_cached_result(explain_module) -> None:
    st_stub = sys.modules["streamlit"]
    st_stub.button.return_value = False

    col_one = MagicMock()
    col_one.__enter__.return_value = col_one
    col_one.__exit__.return_value = False
    col_two = MagicMock()
    col_two.__enter__.return_value = col_two
    col_two.__exit__.return_value = False
    st_stub.columns.return_value = [col_one, col_two]

    run_key = "run:cached"
    cached = explain_module.ExplanationResult(
        text="Cached output",
        trace_url=None,
        claim_issues=[],
        metric_count=1,
        created_at="2024-01-01T00:00:00+00:00",
    )
    st_stub.session_state[explain_module._CACHE_KEY] = {run_key: cached}

    details = {"run_id": "run-001"}
    result = SimpleNamespace(details=details)

    explain_module.render_explain_results(result, run_key=run_key)

    st_stub.markdown.assert_any_call("Cached output")


def test_render_explain_results_reports_llm_error(explain_module, monkeypatch) -> None:
    st_stub = sys.modules["streamlit"]
    st_stub.button.return_value = True

    spinner = MagicMock()
    spinner.__enter__.return_value = spinner
    spinner.__exit__.return_value = False
    st_stub.spinner.return_value = spinner

    def _raise_error(*_args, **_kwargs):
        raise ValueError("Missing API key for openai.")

    monkeypatch.setattr(explain_module, "generate_result_explanation", _raise_error)

    result = SimpleNamespace(details={"out_sample_stats": {"Portfolio": (0.1,)}})

    explain_module.render_explain_results(result, run_key="run:error")

    st_stub.error.assert_any_call("We could not generate an explanation.")
    st_stub.caption.assert_any_call("Missing API key for openai.")
