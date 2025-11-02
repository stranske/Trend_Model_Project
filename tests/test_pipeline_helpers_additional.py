"""Additional unit tests for ``trend_analysis.pipeline`` helper utilities."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

import trend_analysis.pipeline as pipeline
from trend_analysis.pipeline import (
    _build_trend_spec,
    _cfg_section,
    _cfg_value,
    _derive_split_from_periods,
    _policy_from_config,
    _prepare_input_data,
    _preprocessing_summary,
    _resolve_sample_split,
    _section_get,
    _unwrap_cfg,
)
from trend_analysis.util.frequency import FrequencySummary
from trend_analysis.util.missing import MissingPolicyResult


class DummyMapping(Mapping[str, Any]):
    """Mapping-like object exposing a ``get`` method with optional default."""

    def __init__(self, data: Mapping[str, Any]):
        self._data = dict(data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):  # pragma: no cover - mapping protocol
        return iter(self._data)

    def __len__(self):  # pragma: no cover - mapping protocol
        return len(self._data)

    def get(self, key: str, default: Any = None):
        if default is ...:
            raise TypeError("Unexpected ellipsis default")
        return self._data.get(key, default)


def test_cfg_helpers_handle_mixed_inputs() -> None:
    cfg = SimpleNamespace(alpha=1, beta={"nested": 2})
    mapping = {"gamma": 3}

    assert _cfg_value(cfg, "alpha") == 1
    assert _cfg_value(mapping, "gamma") == 3
    assert _cfg_value(mapping, "missing", 5) == 5

    section = _cfg_section(cfg, "beta")
    assert isinstance(section, dict)
    assert section["nested"] == 2

    wrapped = {"__cfg__": {"delta": 4}}
    unwrapped = _unwrap_cfg(wrapped)
    assert unwrapped == {"delta": 4}

    nested_mapping = DummyMapping({"value": 10})
    assert _section_get(nested_mapping, "value") == 10
    assert _section_get(nested_mapping, "other", default=7) == 7


def test_preprocessing_summary_includes_missing_info() -> None:
    summary = _preprocessing_summary(
        "D", normalised=True, missing_summary="ffill applied"
    )
    assert "Cadence: Daily" in summary
    assert "monthly" in summary.lower()
    assert "ffill applied" in summary

    summary_month = _preprocessing_summary("M", normalised=False, missing_summary=None)
    assert "(month-end)" in summary_month


def test_build_trend_spec_uses_vol_adjust_defaults() -> None:
    cfg = {
        "signals": {
            "window": "12",
            "min_periods": "4",
            "lag": "0",
            "vol_target": "0.15",
            "zscore": True,
        }
    }
    vol_cfg = {"enabled": True, "target_vol": 0.2}
    spec = _build_trend_spec(cfg, vol_cfg)
    assert spec.window == 12
    assert spec.min_periods == 4
    # Lag coerces to minimum of 1 when invalid.
    assert spec.lag == 1
    # ``vol_adjust`` should fall back to the vol config default when unset.
    assert spec.vol_adjust is True
    assert spec.vol_target == pytest.approx(0.15)
    assert spec.zscore is True


def test_policy_from_config_constructs_composites() -> None:
    policy, limit = _policy_from_config(
        {
            "policy": "drop",
            "per_asset": {"A": "ffill", "B": "zero"},
            "limit": 2,
            "per_asset_limit": {"A": 1},
        }
    )
    assert isinstance(policy, dict)
    assert policy["default"] == "drop"
    assert policy["B"] == "zero"
    assert isinstance(limit, dict)
    assert limit["default"] == 2
    assert limit["A"] == 1


def test_derive_split_ratio_fallback_when_date_split_invalid() -> None:
    periods = pd.period_range("2020-01", periods=6, freq="M")
    result = _derive_split_from_periods(
        periods,
        method="date",
        boundary=pd.Period("2019-12", freq="M"),
        ratio=0.4,
    )
    # Boundary precedes all periods; ratio fallback should apply.
    assert result["in_end"] == "2020-02"
    assert result["out_start"] == "2020-03"

    ratio_edge = _derive_split_from_periods(
        periods[:2], method="ratio", boundary=None, ratio=1.5
    )
    # Even with an extreme ratio the helper leaves at least one out-of-sample period.
    assert ratio_edge["out_start"] == "2020-02"


def test_resolve_sample_split_derives_missing_fields() -> None:
    df = pd.DataFrame(
        {"Date": pd.date_range("2020-01-31", periods=6, freq="ME"), "A": np.arange(6)}
    )
    split_cfg = {"method": "date", "date": "2020-03"}
    resolved = _resolve_sample_split(df, split_cfg)
    assert resolved["in_end"] == "2020-03"
    assert resolved["out_start"] == "2020-04"

    ratio_cfg = {"method": "ratio", "ratio": "0.5"}
    resolved_ratio = _resolve_sample_split(df, ratio_cfg)
    assert resolved_ratio["in_start"] == "2020-01"
    assert resolved_ratio["out_start"] == "2020-04"

    with pytest.raises(ValueError, match="'Date'"):
        _resolve_sample_split(pd.DataFrame({"value": [1, 2]}), {})


def test_prepare_input_data_resamples_and_applies_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=4, freq="D"),
            "A": [0.01, 0.02, 0.03, 0.04],
            "B": [0.0, 0.01, 0.0, 0.02],
        }
    )

    summary = FrequencySummary(
        code="D", label="Daily", resampled=True, target="M", target_label="Monthly"
    )
    fake_result = MissingPolicyResult(
        policy={"A": "drop", "B": "drop"},
        default_policy="drop",
        limit={"A": None, "B": None},
        default_limit=None,
        filled={"A": 2, "B": 1},
        dropped_assets=(),
        summary="ffill 2 rows",
    )

    with monkeypatch.context() as mp:
        mp.setattr(pipeline, "detect_frequency", lambda series: summary)

        def fake_apply_missing_policy(data: pd.DataFrame, **kwargs):
            assert kwargs["policy"] == "drop"
            assert kwargs["enforce_completeness"] is True
            return data, fake_result

        mp.setattr(pipeline, "apply_missing_policy", fake_apply_missing_policy)

        monthly, freq_info, missing_meta, normalised = _prepare_input_data(
            df,
            date_col="Date",
            missing_policy="drop",
            missing_limit=None,
            enforce_completeness=True,
        )

    assert normalised is True
    assert isinstance(freq_info, FrequencySummary)
    assert missing_meta is fake_result
    assert list(monthly.columns) == ["Date", "A", "B"]
    # Resampled monthly DataFrame should collapse to the end of the month.
    assert all(monthly["Date"].dt.is_month_end)
