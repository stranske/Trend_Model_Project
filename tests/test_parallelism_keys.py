"""Regression tests for canonical parallelism config keys."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from trend.spec import load_run_spec_from_mapping
from trend_analysis.config.schema_generator import generate_schema
from trend_analysis.config.schema_validation import validate_config_data


def _defaults_payload() -> dict:
    payload = yaml.safe_load(Path("config/defaults.yml").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_run_n_jobs_absent_from_defaults_and_schema() -> None:
    payload = _defaults_payload()
    assert "n_jobs" not in payload["run"]

    schema = generate_schema()
    run_properties = schema["properties"]["run"]["properties"]
    assert "n_jobs" not in run_properties
    assert run_properties["jobs"]["type"] == ["integer", "null"]
    assert "jobs" not in schema["properties"]

    errors = validate_config_data({"run": {"n_jobs": 1}}, schema)
    assert errors
    assert any("n_jobs" in error for error in errors)


def test_run_jobs_is_canonical_and_top_level_jobs_is_rejected() -> None:
    payload = _defaults_payload()
    assert "jobs" not in payload

    canonical_payload = copy.deepcopy(payload)
    canonical_payload["run"]["jobs"] = 3
    canonical_spec = load_run_spec_from_mapping(canonical_payload, base_path=Path("config"))
    assert canonical_spec.backtest.jobs == 3

    alias_payload = copy.deepcopy(payload)
    alias_payload["jobs"] = 7
    errors = validate_config_data(alias_payload, generate_schema())
    assert errors
    assert any("jobs" in error for error in errors)
