"""Regression coverage for ``trend_analysis.signal_presets`` helpers."""

from __future__ import annotations

import importlib

from trend_analysis import signal_presets as module
from trend_analysis.signal_presets import TrendSpecPreset
from trend_analysis.signals import TrendSpec


def test_resolve_trend_spec_accepts_empty_string() -> None:
    """Empty string inputs should resolve to the default preset."""

    default = module.get_trend_spec_preset(module.default_preset_name())
    assert module.resolve_trend_spec("") is default


def test_ordered_presets_helpers_are_sorted() -> None:
    """Helper generators should expose presets sorted by their keys."""

    items = list(module._ordered_presets_items())  # type: ignore[attr-defined]
    keys = [key for key, _ in items]
    assert keys == sorted(keys)

    ordered = list(module._ordered_presets())  # type: ignore[attr-defined]
    assert [preset.name for preset in ordered] == module.list_trend_spec_presets()


def test_payload_helpers_preserve_explicit_values() -> None:
    """Explicit numeric values must survive both config and form helpers."""

    explicit_preset = TrendSpecPreset(
        name="Explicit",
        description="",
        spec=TrendSpec(
            window=20,
            min_periods=5,
            lag=2,
            vol_adjust=True,
            vol_target=0.0,
            zscore=False,
        ),
    )

    config = explicit_preset.as_signal_config()
    assert config["min_periods"] == 5
    assert config["vol_target"] == 0.0

    defaults = explicit_preset.form_defaults()
    assert defaults["min_periods"] == 5
    assert defaults["vol_target"] == 0.0


def test_module_all_exports_match_public_api() -> None:
    """``__all__`` should expose the documented public helpers."""

    reloaded = importlib.reload(module)
    exported = {getattr(reloaded, name) for name in reloaded.__all__}
    assert module.TrendSpecPreset in exported
    assert module.default_preset_name in exported
    assert module.list_trend_spec_presets in exported
    assert module.list_trend_spec_keys in exported
    assert module.get_trend_spec_preset in exported
    assert module.resolve_trend_spec in exported

    assert "_ordered_presets" not in reloaded.__all__
    assert "_ordered_presets_items" not in reloaded.__all__
