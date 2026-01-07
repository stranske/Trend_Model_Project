"""Tests for schema selection helpers."""

from __future__ import annotations

from trend_analysis.llm.schema import select_schema_sections


def test_select_schema_sections_filters_to_relevant_top_level_keys() -> None:
    schema = {
        "type": "object",
        "default": {"portfolio": {"constraints": {"max_weight": 0.2}}, "data": {"currency": "USD"}},
        "properties": {
            "portfolio": {
                "type": "object",
                "properties": {
                    "constraints": {
                        "type": "object",
                        "properties": {
                            "max_weight": {"type": "number"},
                        },
                    }
                },
            },
            "data": {
                "type": "object",
                "properties": {
                    "currency": {"type": "string"},
                },
            },
        },
    }

    filtered = select_schema_sections(schema, "Increase max weight to 0.3")

    assert "portfolio" in filtered["properties"]
    assert "data" not in filtered["properties"]
    assert filtered["default"] == {"portfolio": {"constraints": {"max_weight": 0.2}}}


def test_select_schema_sections_returns_full_schema_when_no_match() -> None:
    schema = {
        "type": "object",
        "properties": {
            "portfolio": {"type": "object"},
            "data": {"type": "object"},
        },
    }

    filtered = select_schema_sections(schema, "Unknown instruction")

    assert filtered == schema
