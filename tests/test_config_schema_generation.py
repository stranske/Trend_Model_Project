"""Tests for config schema generation and validation helpers."""

from __future__ import annotations

from trend_analysis.config.schema_generator import generate_schema
from trend_analysis.config.schema_validation import validate_config_data


def _walk_schema(schema: dict) -> list[dict]:
    nodes = []

    def _walk(node: dict) -> None:
        nodes.append(node)
        if node.get("type") == "object" and "properties" in node:
            for child in node["properties"].values():
                _walk(child)

    _walk(schema)
    return nodes


def test_schema_includes_metadata() -> None:
    schema = generate_schema()
    nodes = _walk_schema(schema)

    # Exclude the root node from the key count.
    key_nodes = nodes[1:]
    assert len(key_nodes) >= 30
    for node in key_nodes:
        assert "type" in node
        assert "description" in node
        assert "default" in node
        assert "constraints" in node
        assert "nl_editable" in node


def test_schema_validation_flags_unknown_keys() -> None:
    schema = generate_schema()
    errors = validate_config_data({"unknown_key": 1}, schema)
    assert errors
    assert any("unknown_key" in error for error in errors)
