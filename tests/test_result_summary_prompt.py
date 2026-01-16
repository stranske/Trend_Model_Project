"""Tests for analysis result summary prompt template."""

from __future__ import annotations

from trend_analysis.llm.prompts import (
    DEFAULT_RESULT_RULES,
    DEFAULT_RESULT_SYSTEM_PROMPT,
    SECTION_RESULT_METRICS,
    SECTION_RESULT_OUTPUT,
    SECTION_RESULT_QUESTIONS,
    SECTION_RESULT_RULES,
    SECTION_RESULT_SYSTEM,
    build_result_summary_prompt,
)


def test_build_result_summary_prompt_includes_required_sections() -> None:
    prompt = build_result_summary_prompt(
        analysis_output="Out-sample stats: CAGR 8%.",
        metric_catalog="out_sample_stats.cagr: 0.08",
        questions="Summarize performance.",
    )

    assert f"## {SECTION_RESULT_SYSTEM}" in prompt
    assert f"## {SECTION_RESULT_OUTPUT}" in prompt
    assert f"## {SECTION_RESULT_METRICS}" in prompt
    assert f"## {SECTION_RESULT_RULES}" in prompt
    assert f"## {SECTION_RESULT_QUESTIONS}" in prompt


def test_result_rules_and_disclaimer_are_present() -> None:
    prompt = build_result_summary_prompt(
        analysis_output="Out-sample stats: CAGR 8%.",
        metric_catalog="out_sample_stats.cagr: 0.08",
        questions="Summarize performance.",
    )

    for rule in DEFAULT_RESULT_RULES:
        assert rule in prompt

    assert "This is analytical output, not financial advice." in DEFAULT_RESULT_SYSTEM_PROMPT
    assert "This is analytical output, not financial advice." in prompt
