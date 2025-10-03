from types import SimpleNamespace

import pytest

from trend_analysis.presets import (
    apply_trend_preset,
    get_trend_preset,
    list_preset_slugs,
)


def test_preset_registry_includes_expected_slugs():
    slugs = list_preset_slugs()
    assert "conservative" in slugs
    assert "aggressive" in slugs


@pytest.mark.parametrize("name", ["conservative", "Conservative"])
def test_get_trend_preset_resolves_case_insensitive(name):
    preset = get_trend_preset(name)
    assert preset.slug == "conservative"
    assert preset.trend_spec.window == 126
    assert pytest.approx(preset.trend_spec.vol_target or 0.0, rel=1e-6) == 0.08


def test_apply_trend_preset_updates_config():
    preset = get_trend_preset("aggressive")
    cfg = SimpleNamespace(vol_adjust={}, run={}, signals={})

    apply_trend_preset(cfg, preset)

    assert cfg.signals["window"] == preset.trend_spec.window
    assert cfg.signals["vol_adjust"] is True
    assert pytest.approx(cfg.signals["vol_target"], rel=1e-6) == 0.15
    assert cfg.vol_adjust["window"]["length"] == preset.trend_spec.window
    assert cfg.run["trend_preset"] == "aggressive"
