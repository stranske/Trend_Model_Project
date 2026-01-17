"""Tests for result summary chain handling of unavailable metrics."""

from __future__ import annotations

import pytest

from trend_analysis.llm.chain import ResultSummaryChain
from trend_analysis.llm.result_metrics import MetricEntry


def test_result_summary_chain_rejects_unavailable_metric_requests() -> None:
    def _prompt_builder(**_kwargs: object) -> str:
        raise AssertionError("prompt builder should not be called for missing metrics")

    chain = ResultSummaryChain(llm=object(), prompt_builder=_prompt_builder)
    entries = [
        MetricEntry(
            path="out_sample_stats.Portfolio.cagr",
            value=0.12,
            source="out_sample_stats",
        )
    ]

    response = chain.run(
        analysis_output="summary table",
        metric_catalog="catalog",
        questions="Report alpha and beta for the strategy.",
        metric_entries=entries,
    )

    assert "Requested data is unavailable in the analysis output for: alpha, beta." in response.text
    assert "This is analytical output, not financial advice." in response.text
