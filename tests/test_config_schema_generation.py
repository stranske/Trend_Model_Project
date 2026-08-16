"""Tests for config schema generation and validation helpers."""

from __future__ import annotations

from trend_analysis.config_contract import SUPPORTED_PORTFOLIO_WEIGHTING_NAMES
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


def test_weighting_enum_uses_runtime_supported_names() -> None:
    schema = generate_schema()
    weighting_name = schema["properties"]["portfolio"]["properties"]["weighting"]["properties"][
        "name"
    ]

    assert set(weighting_name["enum"]) == set(SUPPORTED_PORTFOLIO_WEIGHTING_NAMES)


def test_schema_validation_flags_unknown_keys() -> None:
    schema = generate_schema()
    errors = validate_config_data({"unknown_key": 1}, schema)
    assert errors
    assert any("unknown_key" in error for error in errors)


def test_schema_validation_accepts_regime_turnover_caps() -> None:
    schema = generate_schema()
    payload = {"portfolio": {"max_turnover": {"risk_on": 0.2, "risk_off": 0.1}}}
    errors = validate_config_data(payload, schema)
    assert errors == []


def test_schema_validation_accepts_rf_override_enabled() -> None:
    schema = generate_schema()
    metrics_schema = schema["properties"]["metrics"]["properties"]["rf_override_enabled"]
    assert metrics_schema["type"] == "boolean"
    assert metrics_schema["default"] is False

    errors = validate_config_data({"metrics": {"rf_override_enabled": True}}, schema)
    assert errors == []


def test_schema_validation_accepts_regime_model() -> None:
    schema = generate_schema()
    regime_schema = schema["properties"]["regime"]["properties"]["model"]
    assert regime_schema["type"] == "string"
    assert regime_schema["default"] == "binary_threshold"

    errors = validate_config_data({"regime": {"model": "binary_threshold"}}, schema)
    assert errors == []


def test_schema_validation_accepts_cost_model_float_bps() -> None:
    schema = generate_schema()
    cost_schema = schema["properties"]["portfolio"]["properties"]["cost_model"]["properties"]
    assert cost_schema["per_trade_bps"]["type"] == "number"
    assert cost_schema["per_trade_bps"]["minimum"] == 0
    assert cost_schema["half_spread_bps"]["type"] == "number"
    assert cost_schema["half_spread_bps"]["minimum"] == 0

    errors = validate_config_data(
        {"portfolio": {"cost_model": {"per_trade_bps": 2.5, "half_spread_bps": 0.75}}},
        schema,
    )
    assert errors == []


def test_schema_validation_rejects_misspelled_rf_override_enabled() -> None:
    schema = generate_schema()
    errors = validate_config_data({"metrics": {"rf_override_enbaled": True}}, schema)
    assert errors
    assert any("metrics" in error and "rf_override_enbaled" in error for error in errors)


def test_schema_validation_rejects_invalid_regime_turnover_caps() -> None:
    schema = generate_schema()
    payload = {"portfolio": {"max_turnover": {"risk_on": "fast"}}}
    errors = validate_config_data(payload, schema)
    assert errors
    assert any("max_turnover" in error for error in errors)
