"""Tests for the variant patch prompt template."""

from __future__ import annotations

from trend_analysis.llm.prompts import (
    SECTION_VARIANT_GUIDELINES,
    build_variant_patch_prompt,
)


def test_variant_prompt_includes_guidelines_section() -> None:
    prompt = build_variant_patch_prompt(
        current_config="portfolio:\n  max_weight: 0.2",
        allowed_schema='{"type": "object"}',
        instruction="Generate three variants with different max_weight.",
    )

    assert f"## {SECTION_VARIANT_GUIDELINES}" in prompt
    assert "conservative:" in prompt
    assert "baseline:" in prompt
    assert "aggressive:" in prompt
    assert "Reduce risk" in prompt
    assert "lower exposure" in prompt
    assert "conservative patch meaningfully lower risk" in prompt
