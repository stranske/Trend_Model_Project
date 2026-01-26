"""Tests for comparison prompt template."""

from __future__ import annotations

from trend_analysis.llm.prompts import (
    DEFAULT_COMPARISON_RULES,
    DEFAULT_COMPARISON_SYSTEM_PROMPT,
    SECTION_COMPARISON_METRICS_A,
    SECTION_COMPARISON_METRICS_B,
    SECTION_COMPARISON_QUESTIONS,
    SECTION_COMPARISON_RESULT_A,
    SECTION_COMPARISON_RESULT_B,
    SECTION_COMPARISON_RULES,
    SECTION_COMPARISON_SYSTEM,
    build_comparison_prompt,
)


def test_build_comparison_prompt_includes_required_sections() -> None:
    prompt = build_comparison_prompt(
        analysis_output=("Sim A output", "Sim B output"),
        metric_catalog=("metrics A", "metrics B"),
        questions="Compare the results.",
    )

    assert f"## {SECTION_COMPARISON_SYSTEM}" in prompt
    assert f"## {SECTION_COMPARISON_RESULT_A}" in prompt
    assert f"## {SECTION_COMPARISON_RESULT_B}" in prompt
    assert f"## {SECTION_COMPARISON_METRICS_A}" in prompt
    assert f"## {SECTION_COMPARISON_METRICS_B}" in prompt
    assert f"## {SECTION_COMPARISON_RULES}" in prompt
    assert f"## {SECTION_COMPARISON_QUESTIONS}" in prompt


def test_comparison_rules_and_disclaimer_are_present() -> None:
    prompt = build_comparison_prompt(
        analysis_output=("Sim A output", "Sim B output"),
        metric_catalog=("metrics A", "metrics B"),
        questions="Compare the results.",
    )

    for rule in DEFAULT_COMPARISON_RULES:
        assert rule in prompt

    assert "This is analytical output, not financial advice." in DEFAULT_COMPARISON_SYSTEM_PROMPT
    assert "This is analytical output, not financial advice." in prompt
