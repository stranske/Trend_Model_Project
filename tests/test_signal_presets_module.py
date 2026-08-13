"""Focused tests for ``trend_analysis.signal_presets`` helpers."""

from pathlib import Path

import pytest

from trend_analysis import presets as preset_module
from trend_analysis import signal_presets as signal_module
from trend_analysis.signal_presets import (
    TrendSpecPreset,
    default_preset_name,
    get_trend_spec_preset,
    list_trend_spec_keys,
    list_trend_spec_presets,
    resolve_trend_spec,
)
from trend_analysis.signals import TrendSpec


def test_default_and_listing_order() -> None:
    """Listings should expose the presets in deterministic, title-cased order."""

    assert default_preset_name() == "Balanced"
    assert list_trend_spec_presets() == ["Aggressive", "Balanced", "Conservative"]
    assert list_trend_spec_keys() == ["aggressive", "balanced", "conservative"]


def test_get_trend_spec_preset_case_insensitive_and_strips_whitespace() -> None:
    """Preset lookups are case-insensitive and trim stray whitespace."""

    baseline = get_trend_spec_preset("balanced")
    assert baseline is get_trend_spec_preset("  BALANCED  ")

    with pytest.raises(KeyError):
        get_trend_spec_preset("unknown")


def test_signal_view_refreshes_after_registry_payload_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Signal views must follow a reloaded canonical preset payload."""

    preset_dir = tmp_path / "presets"
    preset_dir.mkdir()
    preset_path = preset_dir / "balanced.yml"
    preset_path.write_text(
        "name: Balanced\ndescription: First payload\nsignals:\n  window: 42\n  lag: 1\n"
        "  vol_adjust: false\n  zscore: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(preset_module, "PRESETS_DIR", preset_dir)
    monkeypatch.setattr(preset_module, "_DEFAULT_PRESETS_DIR", tmp_path / "unused-defaults")
    monkeypatch.delenv("TREND_PRESETS_DIR", raising=False)
    preset_module._preset_registry.cache_clear()
    signal_module._signal_view.cache_clear()

    assert get_trend_spec_preset("balanced").description == "First payload"

    preset_path.write_text(
        "name: Balanced\ndescription: Refreshed payload\nsignals:\n  window: 96\n  lag: 1\n"
        "  vol_adjust: false\n  zscore: true\n",
        encoding="utf-8",
    )
    preset_module._preset_registry.cache_clear()

    refreshed = get_trend_spec_preset("balanced")
    assert refreshed.description == "Refreshed payload"
    assert refreshed.spec.window == 96


def test_resolve_trend_spec_returns_default_when_name_missing() -> None:
    """Resolving with falsy names should return the module default preset."""

    default_preset = get_trend_spec_preset(default_preset_name())
    assert resolve_trend_spec(None) is default_preset
    assert resolve_trend_spec("") is default_preset

    aggressive = get_trend_spec_preset("Aggressive")
    assert resolve_trend_spec("Aggressive") is aggressive

    with pytest.raises(KeyError):
        resolve_trend_spec("nonexistent")


def test_trend_spec_preset_payload_helpers_include_optional_fields() -> None:
    """Preset helpers should surface optional values when they are configured."""

    aggressive = get_trend_spec_preset("aggressive")
    config = aggressive.as_signal_config()
    assert config["min_periods"] == aggressive.spec.min_periods
    assert config["vol_target"] == aggressive.spec.vol_target

    defaults = aggressive.form_defaults()
    assert defaults["min_periods"] == aggressive.spec.min_periods
    assert defaults["vol_target"] == aggressive.spec.vol_target


def test_trend_spec_preset_payload_helpers_omit_optional_fields_when_absent() -> None:
    """Custom presets without optional values should omit them from mappings."""

    custom = TrendSpecPreset(
        name="Custom",
        description="Custom description",
        spec=TrendSpec(
            window=12,
            min_periods=None,
            lag=3,
            vol_adjust=False,
            vol_target=None,
            zscore=True,
        ),
    )

    config = custom.as_signal_config()
    assert config == {
        "kind": "tsmom",
        "window": 12,
        "lag": 3,
        "vol_adjust": False,
        "zscore": True,
    }

    defaults = custom.form_defaults()
    assert defaults == {
        "window": 12,
        "min_periods": 0,
        "lag": 3,
        "vol_adjust": False,
        "vol_target": 0.0,
        "zscore": True,
    }
