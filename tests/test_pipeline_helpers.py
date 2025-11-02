"""Focused unit tests for helper utilities in ``trend_analysis.pipeline``."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from trend_analysis.pipeline import (
    _cfg_section,
    _cfg_value,
    _derive_split_from_periods,
    _frequency_label,
    _policy_from_config,
    _prepare_input_data,
    _resolve_sample_split,
    _section_get,
    _unwrap_cfg,
)
from trend_analysis.util.frequency import FrequencySummary


def test_cfg_helpers_handle_mappings_and_objects() -> None:
    section = {"key": "value"}
    assert _cfg_section({"section": section}, "section") == section
    assert _cfg_section(SimpleNamespace(section=section), "section") == section
    assert _cfg_value(SimpleNamespace(attr=1), "attr") == 1

    class Fallback:
        def get(self, key: str, default: object = None) -> object:
            if key == "present":
                return "yes"
            raise KeyError

    assert _section_get(Fallback(), "present") == "yes"
    assert _section_get({}, "missing", default="fallback") == "fallback"


def test_unwrap_cfg_flattens_nested_mappings() -> None:
    nested = {"__cfg__": {"__cfg__": {"value": 42}}}
    assert _unwrap_cfg(nested) == {"value": 42}


def test_policy_from_config_merges_defaults() -> None:
    spec = {
        "policy": "drop",
        "per_asset": {"A": "zero"},
        "limit": 5,
        "per_asset_limit": {"A": 2},
    }
    policy, limit = _policy_from_config(spec)
    assert policy == {"default": "drop", "A": "zero"}
    assert limit == {"default": 5, "A": 2}


def test_frequency_label_known_codes() -> None:
    assert _frequency_label("M") == "Monthly"
    assert _frequency_label("D") == "Daily"
    assert _frequency_label("Q") == "Q"


def test_derive_split_from_periods_handles_ratio_and_date() -> None:
    periods = pd.period_range("2020-01", periods=6, freq="M")
    result = _derive_split_from_periods(
        periods, method="date", boundary=pd.Period("2020-03", freq="M"), ratio=0.5
    )
    assert result["in_end"] == "2020-03"
    fallback = _derive_split_from_periods(
        periods, method="ratio", boundary=None, ratio=0.2
    )
    assert fallback["in_end"] == "2020-01"


def test_resolve_sample_split_validates_dataframe() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=4, freq="M"),
            "Fund": [0.1, 0.2, 0.3, 0.4],
        }
    )
    config = {"method": "date", "date": "2020-02", "ratio": 0.5}
    split = _resolve_sample_split(df, config)
    assert split["in_start"] == "2020-01"

    with pytest.raises(ValueError):
        _resolve_sample_split(pd.DataFrame({"Fund": [0.1]}), {})


def test_prepare_input_data_applies_missing_policy() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=3, freq="M"),
            "FundA": [0.1, None, 0.2],
            "FundB": [0.05, 0.06, 0.07],
        }
    )
    processed, summary, missing, normalised = _prepare_input_data(
        df,
        date_col="Date",
        missing_policy={"default": "ffill"},
        missing_limit={"default": 1},
    )
    assert isinstance(summary, FrequencySummary)
    assert normalised is False
    assert "FundA" in processed.columns
    assert missing.policy["FundA"] == "ffill"
