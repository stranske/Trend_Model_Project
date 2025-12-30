import pathlib
import sys

import pytest
from hypothesis import given
from hypothesis import strategies as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # noqa: E402

from trend_analysis import config  # noqa: E402

Config = config.models.Config

# Try to import ValidationError, fallback to ValueError for environments without pydantic
ValidationException: type[Exception]
try:
    from pydantic import ValidationError
except ImportError:
    ValidationException = ValueError
else:
    ValidationException = ValidationError

# Use constants from the Config class to avoid hardcoded duplication
_DICT_SECTIONS = Config.REQUIRED_DICT_FIELDS

invalid_values = st.one_of(st.integers(), st.floats(), st.booleans(), st.lists(st.integers()))


def _base_cfg() -> dict[str, object]:
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


@given(field=st.sampled_from(_DICT_SECTIONS), val=invalid_values)
def test_sections_require_mappings(field, val):
    cfg = _base_cfg()
    cfg[field] = val
    with pytest.raises((ValidationException, ValueError, TypeError)):
        config.load_config(cfg)


@given(val=invalid_values)
def test_version_must_be_string(val):
    cfg = _base_cfg()
    cfg["version"] = val
    with pytest.raises((ValidationException, ValueError, TypeError)):
        config.load_config(cfg)


def test_config_field_constants_synchronized():
    """Test that field constants are synchronized across all Config
    implementations."""
    # Ensure both Pydantic and fallback modes have the same field constants
    pydantic_required = config.models.Config.REQUIRED_DICT_FIELDS
    pydantic_all = config.models.Config.ALL_FIELDS

    # Test that the constants contain expected core fields
    expected_dict_fields = {
        "data",
        "preprocessing",
        "vol_adjust",
        "sample_split",
        "portfolio",
        "metrics",
        "export",
        "run",
    }
    assert set(pydantic_required) == expected_dict_fields

    # Test that ALL_FIELDS contains REQUIRED_DICT_FIELDS plus other fields
    assert set(pydantic_required).issubset(set(pydantic_all))

    # Ensure PresetConfig uses compatible field list
    preset_fields = set(config.models.PresetConfig.PRESET_DICT_FIELDS)
    assert preset_fields == expected_dict_fields


def test_config_constants_match_model_attributes():
    """Test that the field constants match the actual model attributes."""
    from trend_analysis.config.models import Config

    # Create a sample config to test all fields exist as attributes
    cfg_data = _base_cfg()
    cfg_data["version"] = "test"

    cfg = config.load_config(cfg_data)

    # Verify all fields in ALL_FIELDS exist as attributes
    for field_name in Config.ALL_FIELDS:
        assert hasattr(cfg, field_name), f"Field '{field_name}' missing from Config model"

    # Verify all REQUIRED_DICT_FIELDS are actually dicts in defaults
    for field_name in Config.REQUIRED_DICT_FIELDS:
        default_value = getattr(cfg, field_name)
        assert isinstance(default_value, dict), f"Field '{field_name}' should default to dict"
