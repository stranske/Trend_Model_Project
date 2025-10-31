from __future__ import annotations

import pytest

from trend_analysis.signals import TrendSpec
from trend_analysis.signal_presets import (
    TrendSpecPreset,
    default_preset_name,
    get_trend_spec_preset,
    list_trend_spec_keys,
    list_trend_spec_presets,
    resolve_trend_spec,
)


def test_list_trend_spec_presets_contains_expected_names() -> None:
    names = list_trend_spec_presets()
    assert {"Conservative", "Balanced", "Aggressive"}.issubset(set(names))


def test_get_trend_spec_preset_case_insensitive() -> None:
    preset = get_trend_spec_preset("conservative")
    assert isinstance(preset, TrendSpecPreset)
    assert preset.spec.window == 126
    assert preset.spec.vol_adjust is True
    assert pytest.approx(preset.spec.vol_target or 0.0, rel=1e-6) == 0.08


def test_resolve_trend_spec_falls_back_to_default() -> None:
    fallback = resolve_trend_spec(None)
    assert fallback.name == "Balanced"
    config = fallback.as_signal_config()
    assert config["window"] == fallback.spec.window
    assert config["vol_adjust"] is True
    assert config["lag"] == 1


def test_unknown_preset_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_trend_spec_preset("missing")


def test_as_signal_config_handles_optional_fields() -> None:
    preset = TrendSpecPreset(
        name="Custom",
        description="",
        spec=TrendSpec(
            window=40,
            min_periods=None,
            lag=2,
            vol_adjust=False,
            vol_target=None,
            zscore=False,
        ),
    )

    config = preset.as_signal_config()

    assert config == {
        "kind": "tsmom",
        "window": 40,
        "lag": 2,
        "vol_adjust": False,
        "zscore": False,
    }


def test_form_defaults_returns_expected_payload() -> None:
    preset = TrendSpecPreset(
        name="Aggressive",
        description="",
        spec=TrendSpec(
            window=10,
            min_periods=None,
            lag=1,
            vol_adjust=True,
            vol_target=None,
            zscore=True,
        ),
    )

    defaults = preset.form_defaults()

    assert defaults == {
        "window": 10,
        "min_periods": 0,
        "lag": 1,
        "vol_adjust": True,
        "vol_target": 0.0,
        "zscore": True,
    }


def test_default_and_keys_helpers_are_consistent() -> None:
    assert default_preset_name() == "Balanced"
    keys = list_trend_spec_keys()
    assert keys == sorted(keys)
    # Ensure round-trip from keys back to presets preserves ordering
    presets = [resolve_trend_spec(key) for key in keys]
    assert [preset.name for preset in presets] == list_trend_spec_presets()


def test_resolve_trend_spec_is_whitespace_tolerant() -> None:
    aggressive = resolve_trend_spec("  Aggressive  ")
    assert aggressive.name == "Aggressive"
    # When optional fields are present they should surface in configs
    config = aggressive.as_signal_config()
    assert config["min_periods"] == aggressive.spec.min_periods
    assert config["vol_target"] == aggressive.spec.vol_target
