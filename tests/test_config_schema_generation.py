"""Tests for config schema generation and validation helpers."""

from __future__ import annotations

import json
from pathlib import Path

from trend_analysis.config.schema_generator import _compact_schema, generate_schema
from trend_analysis.config.schema_validation import validate_config_data

_COST_MODEL = {"per_trade_bps": 0, "half_spread_bps": 0}


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


def test_weighting_name_remains_open_to_registered_plugins() -> None:
    schema = generate_schema()
    weighting_name = schema["properties"]["portfolio"]["properties"]["weighting"]["properties"][
        "name"
    ]

    assert weighting_name["type"] == "string"
    assert "enum" not in weighting_name
    assert (
        validate_config_data(
            {
                "portfolio": {
                    "cost_model": _COST_MODEL,
                    "weighting": {"name": "third_party_weight_engine"},
                }
            },
            schema,
        )
        == []
    )


def test_defaults_declare_portfolio_weighting_once() -> None:
    portfolio_defaults = generate_schema()["properties"]["portfolio"]["properties"]
    defaults = portfolio_defaults["weighting"]
    source = Path("config/defaults.yml").read_text(encoding="utf-8")

    assert source.count("\n  weighting:\n") == 1
    assert defaults["properties"]["name"]["default"] == "equal"
    assert portfolio_defaults["constraints"]["properties"]["min_weight_strikes"]["default"] == 2


def test_robust_demo_declares_portfolio_weighting_once() -> None:
    source = Path("config/robust_demo.yml").read_text(encoding="utf-8")

    assert source.count("\n  weighting:\n") == 1


def test_checked_in_schemas_match_generator() -> None:
    generated = generate_schema()

    assert json.loads(Path("config.schema.json").read_text(encoding="utf-8")) == generated
    assert json.loads(Path("config.schema.compact.json").read_text(encoding="utf-8")) == (
        _compact_schema(generated)
    )


def test_schema_matches_volatility_window_runtime_bounds() -> None:
    schema = generate_schema()
    window_schema = schema["properties"]["vol_adjust"]["properties"]["window"]
    window = window_schema["properties"]

    assert window_schema["required"] == ["length"]
    assert window["length"]["minimum"] == 1
    assert window["lambda"]["exclusiveMinimum"] == 0
    assert window["lambda"]["exclusiveMaximum"] == 1

    compact_window = _compact_schema(schema)["properties"]["vol_adjust"]["properties"]["window"]
    assert compact_window["required"] == ["length"]
    assert compact_window["properties"]["length"]["minimum"] == 1
    assert compact_window["properties"]["lambda"]["exclusiveMinimum"] == 0
    assert compact_window["properties"]["lambda"]["exclusiveMaximum"] == 1


def test_identity_schema_matches_non_null_runtime_model() -> None:
    schema = generate_schema()

    assert (
        validate_config_data({"identity": {}, "portfolio": {"cost_model": _COST_MODEL}}, schema)
        == []
    )
    errors = validate_config_data(
        {"identity": None, "portfolio": {"cost_model": _COST_MODEL}}, schema
    )
    assert errors
    assert any("identity" in error for error in errors)


def test_schema_validation_flags_unknown_keys() -> None:
    schema = generate_schema()
    errors = validate_config_data(
        {"unknown_key": 1, "portfolio": {"cost_model": _COST_MODEL}}, schema
    )
    assert errors
    assert any("unknown_key" in error for error in errors)


def test_schema_validation_accepts_regime_turnover_caps() -> None:
    schema = generate_schema()
    payload = {
        "portfolio": {
            "cost_model": _COST_MODEL,
            "max_turnover": {"risk_on": 0.2, "risk_off": 0.1},
        }
    }
    errors = validate_config_data(payload, schema)
    assert errors == []


def test_schema_validation_accepts_rf_override_enabled() -> None:
    schema = generate_schema()
    metrics_schema = schema["properties"]["metrics"]["properties"]["rf_override_enabled"]
    assert metrics_schema["type"] == "boolean"
    assert metrics_schema["default"] is False

    errors = validate_config_data(
        {
            "metrics": {"rf_override_enabled": True},
            "portfolio": {"cost_model": _COST_MODEL},
        },
        schema,
    )
    assert errors == []


def test_schema_validation_accepts_regime_model() -> None:
    schema = generate_schema()
    regime_schema = schema["properties"]["regime"]["properties"]["model"]
    assert regime_schema["type"] == "string"
    assert regime_schema["default"] == "binary_threshold"

    errors = validate_config_data(
        {
            "regime": {"model": "binary_threshold"},
            "portfolio": {"cost_model": _COST_MODEL},
        },
        schema,
    )
    assert errors == []


def test_schema_validation_accepts_canonical_signals_section() -> None:
    schema = generate_schema()
    signals_schema = schema["properties"]["signals"]
    payload = {
        "signals": {
            "kind": "tsmom",
            "window": 63,
            "lag": 1,
            "min_periods": None,
            "zscore": False,
            "vol_adjust": False,
            "vol_target": 0.10,
        },
        "portfolio": {"cost_model": _COST_MODEL},
    }

    assert signals_schema["default"] == {}
    assert set(signals_schema["properties"]) == set(payload["signals"])
    assert validate_config_data(payload, schema) == []


def test_schema_rejects_non_positive_numeric_signal_zscore() -> None:
    schema = generate_schema()
    zscore_schema = schema["properties"]["signals"]["properties"]["zscore"]

    assert zscore_schema["exclusiveMinimum"] == 0
    for value in (True, False, 1.5):
        assert (
            validate_config_data(
                {
                    "signals": {"zscore": value},
                    "portfolio": {"cost_model": _COST_MODEL},
                },
                schema,
            )
            == []
        )
    for value in (0, -0.5):
        assert validate_config_data(
            {
                "signals": {"zscore": value},
                "portfolio": {"cost_model": _COST_MODEL},
            },
            schema,
        )


def test_schema_validation_accepts_cost_model_float_bps() -> None:
    schema = generate_schema()
    portfolio_schema = schema["properties"]["portfolio"]
    cost_model_schema = portfolio_schema["properties"]["cost_model"]
    cost_properties = cost_model_schema["properties"]
    assert schema["required"] == ["portfolio"]
    assert portfolio_schema["required"] == ["cost_model"]
    assert cost_model_schema["required"] == ["half_spread_bps", "per_trade_bps"]
    assert cost_properties["per_trade_bps"]["type"] == "number"
    assert cost_properties["per_trade_bps"]["minimum"] == 0
    assert cost_properties["half_spread_bps"]["type"] == "number"
    assert cost_properties["half_spread_bps"]["minimum"] == 0

    errors = validate_config_data(
        {"portfolio": {"cost_model": {"per_trade_bps": 2.5, "half_spread_bps": 0.75}}},
        schema,
    )
    assert errors == []
    assert validate_config_data({}, schema)
    assert validate_config_data({"portfolio": {}}, schema)
    assert validate_config_data({"portfolio": {"cost_model": {"per_trade_bps": 2.5}}}, schema)


def test_schema_validation_accepts_canonical_threshold_controls() -> None:
    schema = generate_schema()
    payload = {
        "portfolio": {
            "cost_model": _COST_MODEL,
            "constraints": {"min_weight_strikes": 2},
            "sticky_add_x": 2,
            "sticky_drop_y": 3,
        }
    }

    assert validate_config_data(payload, schema) == []


def test_schema_validation_accepts_canonical_minimum_tenure() -> None:
    schema = generate_schema()
    tenure_schema = schema["properties"]["portfolio"]["properties"]["min_tenure_n"]

    assert tenure_schema["type"] == "integer"
    assert tenure_schema["minimum"] == 0
    assert tenure_schema["default"] == 0
    assert (
        validate_config_data({"portfolio": {"cost_model": _COST_MODEL, "min_tenure_n": 2}}, schema)
        == []
    )


def test_schema_validation_rejects_misspelled_rf_override_enabled() -> None:
    schema = generate_schema()
    errors = validate_config_data(
        {
            "metrics": {"rf_override_enbaled": True},
            "portfolio": {"cost_model": _COST_MODEL},
        },
        schema,
    )
    assert errors
    assert any("metrics" in error and "rf_override_enbaled" in error for error in errors)


def test_schema_validation_rejects_invalid_regime_turnover_caps() -> None:
    schema = generate_schema()
    payload = {
        "portfolio": {
            "cost_model": _COST_MODEL,
            "max_turnover": {"risk_on": "fast"},
        }
    }
    errors = validate_config_data(payload, schema)
    assert errors
    assert any("max_turnover" in error for error in errors)
