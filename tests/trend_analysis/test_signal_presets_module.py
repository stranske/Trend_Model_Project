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


@pytest.fixture()
def balanced_preset() -> TrendSpecPreset:
    return get_trend_spec_preset("balanced")


def test_list_trend_spec_presets_returns_sorted_titles() -> None:
    names = list_trend_spec_presets()
    assert names == ["Aggressive", "Balanced", "Conservative"]


def test_list_trend_spec_keys_returns_sorted_keys() -> None:
    keys = list_trend_spec_keys()
    assert keys == ["aggressive", "balanced", "conservative"]


def test_get_trend_spec_preset_is_case_insensitive(
    balanced_preset: TrendSpecPreset,
) -> None:
    mixed_case = get_trend_spec_preset("BaLanCed")
    assert mixed_case is balanced_preset


def test_get_trend_spec_preset_unknown_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_trend_spec_preset("missing")


def test_resolve_trend_spec_falls_back_to_default(
    balanced_preset: TrendSpecPreset,
) -> None:
    assert resolve_trend_spec(None) is balanced_preset
    assert resolve_trend_spec("") is balanced_preset
    assert resolve_trend_spec("Conservative").name == "Conservative"


def test_default_preset_name_matches_registry() -> None:
    assert default_preset_name() == "Balanced"


def test_as_signal_config_includes_optional_fields(
    balanced_preset: TrendSpecPreset,
) -> None:
    payload = balanced_preset.as_signal_config()
    assert payload == {
        "kind": "tsmom",
        "window": 84,
        "lag": 1,
        "vol_adjust": True,
        "zscore": True,
        "min_periods": 63,
        "vol_target": 0.10,
    }


def test_as_signal_config_omits_none_fields_and_defaults() -> None:
    spec = TrendSpec(
        window=50,
        min_periods=None,
        lag=2,
        vol_adjust=False,
        vol_target=None,
        zscore=False,
    )
    custom = TrendSpecPreset("Custom", "No optional fields", spec)
    payload = custom.as_signal_config()
    assert payload == {
        "kind": "tsmom",
        "window": 50,
        "lag": 2,
        "vol_adjust": False,
        "zscore": False,
    }


def test_form_defaults_zeroes_optional_values() -> None:
    spec = TrendSpec(
        window=40,
        min_periods=None,
        lag=3,
        vol_adjust=True,
        vol_target=None,
        zscore=True,
    )
    custom = TrendSpecPreset("Custom", "Defaults", spec)
    defaults = custom.form_defaults()
    assert defaults == {
        "window": 40,
        "min_periods": 0,
        "lag": 3,
        "vol_adjust": True,
        "vol_target": 0.0,
        "zscore": True,
    }
