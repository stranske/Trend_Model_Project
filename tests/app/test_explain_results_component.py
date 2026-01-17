"""Tests for the Streamlit explain results helpers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
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
