"""Targeted coverage tests for :mod:`trend_analysis.signal_presets`."""

from __future__ import annotations

import pytest

from trend_analysis.signal_presets import (
    TrendSpecPreset,
    _ordered_presets,
    _ordered_presets_items,
    default_preset_name,
    get_trend_spec_preset,
    list_trend_spec_keys,
    list_trend_spec_presets,
    resolve_trend_spec,
)
from trend_analysis.signals import TrendSpec


def test_default_preset_matches_resolve_fallback() -> None:
    default_name = default_preset_name()
    assert default_name == "Balanced"
    # ``resolve_trend_spec`` should fall back to the same preset when the
    # caller provides a falsey value (None, empty string, whitespace only).
    assert resolve_trend_spec(None).name == default_name
    assert resolve_trend_spec("").name == default_name

    explicit = resolve_trend_spec("Aggressive")
    assert explicit is get_trend_spec_preset("aggressive")


def test_get_trend_spec_preset_is_case_insensitive() -> None:
    balanced = get_trend_spec_preset("BALANCED")
    aggressive = get_trend_spec_preset("Aggressive")
    assert balanced.name == "Balanced"
    assert aggressive.spec.window < balanced.spec.window

    with pytest.raises(KeyError):
        get_trend_spec_preset("unknown")


def test_trend_spec_preset_as_signal_config_includes_optional_fields() -> None:
    preset = get_trend_spec_preset("conservative")
    payload = preset.as_signal_config()
    # Optional fields should be included when the preset defines them.
    assert payload == {
        "kind": "tsmom",
        "window": 126,
        "lag": 1,
        "vol_adjust": True,
        "zscore": True,
        "min_periods": 90,
        "vol_target": 0.08,
    }


def test_trend_spec_preset_methods_handle_missing_optional_fields() -> None:
    spec = TrendSpec(window=21, min_periods=None, lag=2, vol_adjust=False, vol_target=None)
    preset = TrendSpecPreset(name="Custom", description="Spec without optional fields", spec=spec)

    # ``as_signal_config`` should only contain fields that are explicitly set
    # on the underlying ``TrendSpec`` instance.
    assert preset.as_signal_config() == {
        "kind": "tsmom",
        "window": 21,
        "lag": 2,
        "vol_adjust": False,
        "zscore": False,
    }

    # ``form_defaults`` should surface sensible defaults for optional values.
    assert preset.form_defaults() == {
        "window": 21,
        "min_periods": 0,
        "lag": 2,
        "vol_adjust": False,
        "vol_target": 0.0,
        "zscore": False,
    }


def test_ordered_presets_match_public_lists() -> None:
    # ``_ordered_presets_items`` drives both list helpers, so ensure it returns
    # the canonical key ordering and objects expected by callers.
    ordered_items = _ordered_presets_items()
    assert [key for key, _ in ordered_items] == list_trend_spec_keys()
    assert [preset.name for _, preset in ordered_items] == list_trend_spec_presets()

    # The generator wrapper should yield the same objects in the same order.
    assert list(_ordered_presets()) == [preset for _, preset in ordered_items]
