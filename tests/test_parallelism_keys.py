"""Regression tests for canonical parallelism config keys."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from trend_analysis.config.schema_generator import generate_schema
from trend_analysis.config.schema_validation import validate_config_data
from trend.spec import load_run_spec_from_mapping


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
    assert run_properties["jobs"]["type"] == "integer"
    assert "Deprecated" in schema["properties"]["jobs"]["description"]

    errors = validate_config_data({"run": {"n_jobs": 1}}, schema)
    assert errors
    assert any("n_jobs" in error for error in errors)


def test_run_jobs_is_canonical_and_top_level_jobs_is_alias() -> None:
    payload = _defaults_payload()
    assert "jobs" not in payload

    canonical_payload = copy.deepcopy(payload)
    canonical_payload["run"]["jobs"] = 3
    canonical_payload["jobs"] = 7
    canonical_spec = load_run_spec_from_mapping(canonical_payload, base_path=Path("config"))
    assert canonical_spec.backtest.jobs == 3

    alias_payload = copy.deepcopy(payload)
    alias_payload["run"].pop("jobs")
    alias_payload["jobs"] = 7
    alias_spec = load_run_spec_from_mapping(alias_payload, base_path=Path("config"))
    assert alias_spec.backtest.jobs == 7
