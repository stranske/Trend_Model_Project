"""Tests for ConfigPatch prompt template."""

from __future__ import annotations

import json

from trend_analysis.llm.prompts import (
    DEFAULT_SAFETY_RULES,
    SECTION_CONFIG,
    SECTION_SAFETY,
    SECTION_SCHEMA,
    SECTION_SYSTEM,
    SECTION_USER,
    build_config_patch_prompt,
    format_config_for_prompt,
)


def test_build_config_patch_prompt_includes_required_sections() -> None:
    config_text = format_config_for_prompt({"portfolio": {"max_weight": 0.2}})
    schema_text = json.dumps({"type": "object"}, indent=2)
    prompt = build_config_patch_prompt(
        current_config=config_text,
        allowed_schema=schema_text,
        instruction="Increase max_weight to 0.3.",
    )

    assert f"## {SECTION_SYSTEM}" in prompt
    assert f"## {SECTION_CONFIG}" in prompt
    assert f"## {SECTION_SCHEMA}" in prompt
    assert f"## {SECTION_SAFETY}" in prompt
    assert f"## {SECTION_USER}" in prompt


def test_default_safety_rules_are_rendered() -> None:
    config_text = format_config_for_prompt({"portfolio": {"max_weight": 0.2}})
    schema_text = json.dumps({"type": "object"}, indent=2)
    prompt = build_config_patch_prompt(
        current_config=config_text,
        allowed_schema=schema_text,
        instruction="Increase max_weight to 0.3.",
    )

    for rule in DEFAULT_SAFETY_RULES:
        assert rule in prompt
