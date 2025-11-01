from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest

from trend_analysis import presets as preset_mod
from trend_analysis.presets import (
    apply_trend_preset,
    get_trend_preset,
    list_preset_slugs,
)


@pytest.fixture(autouse=True)
def reset_preset_registry(monkeypatch: pytest.MonkeyPatch):
    preset_mod._preset_registry.cache_clear()
    yield
    preset_mod._preset_registry.cache_clear()
    monkeypatch.delenv("TREND_PRESETS_DIR", raising=False)


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


def test_env_override_directory_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override_dir = tmp_path / "override"
    override_dir.mkdir()
    yaml_content = dedent(
        """
        name: Conservative
        description: Override conservative preset
        lookback_months: 48
        rebalance_frequency: "monthly"
        min_track_months: 12
        selection_count: 8
        risk_target: 0.07
        metrics:
          sharpe_ratio: 1.0
        signals:
          window: 10
          lag: 1
          vol_adjust: false
        """
    )
    (override_dir / "conservative.yml").write_text(yaml_content, encoding="utf-8")

    monkeypatch.setenv("TREND_PRESETS_DIR", str(override_dir))

    preset = get_trend_preset("conservative")
    assert preset.trend_spec.window == 10
    assert preset.description == "Override conservative preset"
