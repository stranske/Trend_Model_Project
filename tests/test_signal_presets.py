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


@pytest.mark.parametrize(
    "key,label",
    [
        ("aggressive", "Aggressive"),
        ("balanced", "Balanced"),
        ("conservative", "Conservative"),
    ],
)
def test_builtin_presets_accessible(key: str, label: str) -> None:
    preset = get_trend_spec_preset(key)
    assert preset.name == label
    assert get_trend_spec_preset(label).name == label


def test_default_lists_are_sorted() -> None:
    assert default_preset_name() == "Balanced"
    assert list_trend_spec_keys() == ["aggressive", "balanced", "conservative"]
    assert list_trend_spec_presets() == [
        "Aggressive",
        "Balanced",
        "Conservative",
    ]


def test_resolve_trend_spec_handles_falsy_and_case() -> None:
    fallback = get_trend_spec_preset("balanced")
    assert resolve_trend_spec(None) is fallback
    assert resolve_trend_spec("") is fallback
    assert resolve_trend_spec("  BALANCED  ") is fallback


def test_get_trend_spec_preset_raises_for_unknown() -> None:
    with pytest.raises(KeyError, match="Unknown TrendSpec preset"):
        get_trend_spec_preset("missing")


def test_trend_spec_preset_serialisation_helpers() -> None:
    spec = TrendSpec(
        window=21,
        min_periods=None,
        lag=2,
        vol_adjust=False,
        vol_target=None,
        zscore=True,
    )
    preset = TrendSpecPreset(
        name="Custom",
        description="Example",
        spec=spec,
    )

    assert preset.as_signal_config() == {
        "kind": "tsmom",
        "window": 21,
        "lag": 2,
        "vol_adjust": False,
        "zscore": True,
    }

    defaults = preset.form_defaults()
    assert defaults["window"] == 21
    assert defaults["min_periods"] == 0
    assert defaults["lag"] == 2
    assert defaults["vol_adjust"] is False
    assert defaults["vol_target"] == 0.0


def test_trend_spec_preset_serialisation_includes_optional_fields() -> None:
    spec = TrendSpec(
        window=42, min_periods=21, lag=1, vol_adjust=True, vol_target=0.2, zscore=False
    )
    preset = TrendSpecPreset(name="Custom", description="Example", spec=spec)

    mapping = preset.as_signal_config()
    assert mapping["min_periods"] == 21
    assert mapping["vol_target"] == 0.2

    defaults = preset.form_defaults()
    assert defaults["min_periods"] == 21
    assert defaults["vol_target"] == 0.2
    assert defaults["zscore"] is False
