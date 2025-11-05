"""Additional regression tests for ``trend_analysis.signal_presets``."""

from __future__ import annotations

import pytest

from trend_analysis.signal_presets import (
    TrendSpecPreset,
    default_preset_name,
    get_trend_spec_preset,
    list_trend_spec_keys,
    list_trend_spec_presets,
    resolve_trend_spec,
)
from trend_analysis.signals import TrendSpec


def test_list_trend_spec_presets_orders_by_key_case_insensitive() -> None:
    """Preset listings should follow the canonical key ordering."""

    assert list_trend_spec_presets() == ["Aggressive", "Balanced", "Conservative"]
    assert list_trend_spec_keys() == ["aggressive", "balanced", "conservative"]


def test_get_trend_spec_preset_supports_whitespace_and_case() -> None:
    preset = get_trend_spec_preset("  BALANCED  ")

    assert preset.name == "Balanced"
    assert preset.spec.window == 84


def test_get_trend_spec_preset_unknown_name_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_trend_spec_preset("missing")


def test_resolve_trend_spec_falls_back_to_default() -> None:
    default = resolve_trend_spec("")

    assert default.name == "Balanced"
    assert default_preset_name() == "Balanced"


def test_resolve_trend_spec_uses_explicit_name() -> None:
    aggressive = resolve_trend_spec("aggressive")

    assert aggressive.name == "Aggressive"
    assert aggressive.spec.vol_target == 0.15


def test_as_signal_config_excludes_optional_fields_when_absent() -> None:
    custom_spec = TrendSpec(window=50, lag=2)
    preset = TrendSpecPreset(
        name="Custom",
        description="",
        spec=custom_spec,
    )

    payload = preset.as_signal_config()

    assert payload == {
        "kind": "tsmom",
        "window": 50,
        "lag": 2,
        "vol_adjust": False,
        "zscore": False,
    }


def test_as_signal_config_includes_optional_fields_when_present() -> None:
    preset = get_trend_spec_preset("conservative")

    payload = preset.as_signal_config()

    assert payload["min_periods"] == 90
    assert payload["vol_target"] == 0.08


def test_form_defaults_uses_zero_for_optional_values() -> None:
    custom_spec = TrendSpec(window=20, lag=1, zscore=True)
    preset = TrendSpecPreset(
        name="ZeroDefaults",
        description="",
        spec=custom_spec,
    )

    defaults = preset.form_defaults()

    assert defaults == {
        "window": 20,
        "min_periods": 0,
        "lag": 1,
        "vol_adjust": False,
        "vol_target": 0.0,
        "zscore": True,
    }


def test_form_defaults_preserves_optional_values_when_defined() -> None:
    preset = get_trend_spec_preset("balanced")

    defaults = preset.form_defaults()

    assert defaults["min_periods"] == 63
    assert defaults["vol_target"] == 0.10


