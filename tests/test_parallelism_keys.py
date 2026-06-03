"""Regression coverage for canonical parallelism config keys."""

from __future__ import annotations

import copy

import yaml

from trend_analysis.config.schema_generator import generate_schema
from trend_model.spec import load_run_spec_from_mapping


def _load_defaults() -> dict:
    with open("config/defaults.yml", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


def _config_with_jobs(*, run_jobs: int | None = None, top_level_jobs: int | None = None) -> dict:
    payload = _load_defaults()
    payload["run"].pop("jobs", None)
    payload.pop("jobs", None)
    if run_jobs is not None:
        payload["run"]["jobs"] = run_jobs
    if top_level_jobs is not None:
        payload["jobs"] = top_level_jobs
    return payload


def test_run_n_jobs_absent_from_defaults_and_schema() -> None:
    defaults = _load_defaults()
    assert "n_jobs" not in defaults["run"]

    schema = generate_schema()
    run_properties = schema["properties"]["run"]["properties"]
    assert "n_jobs" not in run_properties
    assert "jobs" in run_properties


def test_run_jobs_and_top_level_jobs_alias_resolve_to_same_worker_count() -> None:
    canonical = load_run_spec_from_mapping(_config_with_jobs(run_jobs=3))
    legacy_alias = load_run_spec_from_mapping(_config_with_jobs(top_level_jobs=3))
    both = load_run_spec_from_mapping(_config_with_jobs(run_jobs=3, top_level_jobs=9))

    assert canonical.backtest.jobs == 3
    assert legacy_alias.backtest.jobs == canonical.backtest.jobs
    assert both.backtest.jobs == canonical.backtest.jobs


def test_top_level_jobs_alias_is_centralized_by_config_loader() -> None:
    payload = _config_with_jobs(top_level_jobs=2)
    spec = load_run_spec_from_mapping(copy.deepcopy(payload))

    assert spec.config.jobs == 2
    assert spec.config.run["jobs"] == 2
