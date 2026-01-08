import copy
from pathlib import Path

import pytest

from trend_analysis.config import models as config_models

_DATA_SECTION = {
    "managers_glob": "data/raw/managers/*.csv",
    "date_column": "Date",
    "frequency": "D",
}

_PORTFOLIO_SECTION = {
    "selection_mode": "all",
    "rebalance_calendar": "NYSE",
    "max_turnover": 1.0,
    "transaction_cost_bps": 0,
}


@pytest.fixture()
def base_config_dict():
    return {
        "version": "1.0",
        "data": dict(_DATA_SECTION),
        "preprocessing": {},
        "vol_adjust": {"target_vol": 0.1},
        "sample_split": {},
        "portfolio": dict(_PORTFOLIO_SECTION),
        "metrics": {},
        "export": {},
        "run": {},
    }


def test_config_defaults_and_model_dump(base_config_dict):
    cfg = config_models.Config(**base_config_dict)

    dump = cfg.model_dump()
    for field in config_models.Config.ALL_FIELDS:
        assert field in dump

    assert dump["version"] == "1.0"
    assert dump["benchmarks"] == {}
    assert dump["output"] is None
    assert dump["seed"] == 42


def test_config_requires_required_sections(base_config_dict):
    bad = copy.deepcopy(base_config_dict)
    bad["metrics"] = None
    with pytest.raises(ValueError, match="metrics section is required"):
        config_models.Config(**bad)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("data", [], "data must be a dictionary"),
        ("preprocessing", 1, "preprocessing must be a dictionary"),
    ],
)
def test_config_requires_dict_sections(base_config_dict, field, value, message):
    bad = copy.deepcopy(base_config_dict)
    bad[field] = value
    with pytest.raises(ValueError, match=message):
        config_models.Config(**bad)


@pytest.mark.parametrize(
    "value, match",
    [
        (123, "version must be a string"),
        ("", "at least 1 character"),
        ("   ", "cannot be empty"),
    ],
)
def test_validate_version_value_errors(value, match):
    with pytest.raises(ValueError, match=match):
        config_models._validate_version_value(value)


def test_validate_version_value_passthrough():
    assert config_models._validate_version_value("1.2.3") == "1.2.3"


def test_load_merges_output_into_export(base_config_dict):
    cfg_map = copy.deepcopy(base_config_dict)
    cfg_map["output"] = {"format": ["csv", "json"], "path": "reports/out/results.xlsx"}

    cfg = config_models.load(cfg_map)
    dump = cfg.model_dump()

    assert dump["export"]["formats"] == ["csv", "json"]
    assert dump["export"]["directory"] == str(Path("reports/out"))
    assert dump["export"]["filename"] == "results.xlsx"


def test_load_config_with_mapping(base_config_dict):
    cfg = config_models.load_config(base_config_dict)
    assert isinstance(cfg, config_models.Config)
    assert cfg.version == "1.0"
