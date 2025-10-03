from __future__ import annotations

import pytest

from trend_analysis.signal_presets import (
    TrendSpecPreset,
    get_trend_spec_preset,
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
