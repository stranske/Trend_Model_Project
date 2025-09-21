import importlib
from typing import Any

import importlib
from typing import Any

import pytest

# Import the module directly via its source path
models = importlib.import_module("trend_analysis.config.models")


# Helper to produce a minimal valid configuration mapping
# Required keys are version, data, preprocessing, vol_adjust, sample_split,
# portfolio, metrics, export, and run


def _sample_config() -> dict[str, Any]:
    return {
        "version": "1",
        "data": {
            "managers_glob": "data/raw/managers/*.csv",
            "date_column": "Date",
            "frequency": "D",
        },
        "preprocessing": {},
        "vol_adjust": {"target_vol": 0.1},
        "sample_split": {},
        "portfolio": {
            "selection_mode": "all",
            "rebalance_calendar": "NYSE",
            "max_turnover": 1.0,
            "transaction_cost_bps": 0,
        },
        "metrics": {},
        "export": {},
        "run": {},
    }


def test_load_config_returns_struct():
    cfg = models.load_config(_sample_config())
    assert isinstance(cfg, models.Config)
    dumped = cfg.model_dump()
    assert dumped["version"] == "1"
    # Ensure required dictionary sections round trip via model_dump
    for key in [
        "data",
        "preprocessing",
        "vol_adjust",
        "sample_split",
        "portfolio",
        "metrics",
        "export",
        "run",
    ]:
        assert key in dumped
        assert isinstance(dumped[key], dict)


def test_missing_required_key_raises():
    cfg_dict = _sample_config()
    cfg_dict.pop("version")
    try:
        from pydantic import ValidationError  # type: ignore[import-not-found]

        err_type = (ValueError, ValidationError)
    except (ImportError, ModuleNotFoundError):
        err_type = (ValueError,)

    with pytest.raises(err_type):
        models.Config(**cfg_dict)


def test_invalid_version_type_raises():
    cfg_dict = _sample_config()
    cfg_dict["version"] = 123  # type: ignore[assignment]
    try:
        from pydantic import ValidationError  # type: ignore[import-not-found]

        err_type = (ValueError, ValidationError)
    except (ImportError, ModuleNotFoundError):
        err_type = (ValueError,)

    with pytest.raises(err_type):
        models.Config(**cfg_dict)
