from __future__ import annotations

import logging
from pathlib import Path
from textwrap import dedent

import pytest

from trend_analysis import presets


@pytest.fixture
def preset_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    base_dir = tmp_path / "primary"
    base_dir.mkdir()
    env_dir = tmp_path / "env"
    env_dir.mkdir()

    monkeypatch.setattr(presets, "PRESETS_DIR", base_dir)
    monkeypatch.setenv("TREND_PRESETS_DIR", str(env_dir))
    presets._preset_registry.cache_clear()
    yield base_dir, env_dir
    presets._preset_registry.cache_clear()


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(dedent(content), encoding="utf-8")


def _build_default_yaml() -> str:
    return """
    name: Alpha Preset
    description: Primary preset description
    lookback_months: 24
    selection_count: 5
    rebalance_frequency: monthly
    metrics:
      sharpe_ratio: 1.5
      max_drawdown: 0.5
    signals:
      window: 50
      min_periods: 10
      lag: 3
      vol_adjust: true
      vol_target: 0.2
      zscore: true
    vol_adjust:
      enabled: true
      target_vol: 0.25
      window:
        short: 10
        long: 63
    """


def _build_override_yaml() -> str:
    return """
    name: Override Label
    description: Override preset description
    lookback_months: 36
    selection_count: 8
    metrics:
      volatility: 2.0
    signals:
      window: 75
      lag: 1
    vol_adjust:
      enabled: false
      target_vol: 0.25
      window:
        short: 12
        long: 63
    """


def test_preset_registry_honours_override_and_warns_on_duplicates(
    preset_environment: tuple[Path, Path], caplog: pytest.LogCaptureFixture
) -> None:
    base_dir, env_dir = preset_environment
    _write_yaml(base_dir / "alpha.yml", _build_default_yaml())
    _write_yaml(env_dir / "alpha.yml", _build_override_yaml())
    _write_yaml(base_dir / "beta.yml", "name: Beta\nsignals:\n  window: 30\n")

    with caplog.at_level(logging.WARNING, logger="trend_analysis.presets"):
        registry = presets._preset_registry()
    assert "Duplicate trend preset slug 'alpha'" in caplog.text

    assert set(registry.keys()) == {"alpha", "beta"}
    override = registry["alpha"]
    assert override.label == "Override Label"
    beta = registry["beta"]
    assert beta.trend_spec.window == 30

    listings = presets.list_trend_presets()
    assert [preset.label for preset in listings] == ["Beta", "Override Label"]
    assert presets.list_preset_slugs() == ("alpha", "beta")


def test_get_trend_preset_supports_slug_and_label(
    preset_environment: tuple[Path, Path],
) -> None:
    base_dir, env_dir = preset_environment
    _write_yaml(base_dir / "alpha.yml", _build_default_yaml())
    _write_yaml(env_dir / "alpha.yml", _build_override_yaml())

    preset = presets.get_trend_preset("alpha")
    assert preset.slug == "alpha"
    assert presets.get_trend_preset("override label").slug == "alpha"
    with pytest.raises(KeyError):
        presets.get_trend_preset("")


def test_trend_preset_helpers_produce_expected_defaults(
    preset_environment: tuple[Path, Path],
) -> None:
    base_dir, env_dir = preset_environment
    _write_yaml(base_dir / "alpha.yml", _build_default_yaml())
    _write_yaml(env_dir / "alpha.yml", _build_override_yaml())

    preset = presets.get_trend_preset("alpha")
    defaults = preset.form_defaults()
    assert defaults["lookback_months"] == 36  # override applied
    assert defaults["selection_count"] == 8
    assert defaults["metrics"] == {"vol": 2.0}

    signals = preset.signals_mapping()
    assert signals == {
        "kind": "tsmom",
        "window": 75,
        "lag": 1,
        "vol_adjust": False,
        "zscore": False,
    }

    vol_defaults = preset.vol_adjust_defaults()
    assert vol_defaults["enabled"] is False
    assert vol_defaults["target_vol"] == 0.25
    assert vol_defaults["window"] == {"short": 12, "long": 63, "length": 75}

    metrics = preset.metrics_pipeline()
    assert metrics == {"Volatility": 2.0}


def test_apply_trend_preset_merges_into_config(
    preset_environment: tuple[Path, Path],
) -> None:
    base_dir, env_dir = preset_environment
    _write_yaml(base_dir / "alpha.yml", _build_default_yaml())
    _write_yaml(env_dir / "alpha.yml", _build_override_yaml())

    preset = presets.get_trend_preset("alpha")

    class DummyConfig:
        def __init__(self) -> None:
            self.signals = {"kind": "tsmom", "lag": 5}
            self.vol_adjust = {"enabled": True, "window": {"legacy": 20}}
            self.run = {}

    config = DummyConfig()
    presets.apply_trend_preset(config, preset)

    assert config.signals["window"] == 75
    assert config.signals["vol_adjust"] is False
    assert config.vol_adjust["enabled"] is False
    assert config.vol_adjust["target_vol"] == 0.25
    assert config.vol_adjust["window"] == {"short": 12, "long": 63, "length": 75}
    assert config.run["trend_preset"] == "alpha"


def test_metric_alias_helpers_normalise_inputs() -> None:
    assert presets.normalise_metric_key("Sharpe_Ratio") == "sharpe"
    assert presets.normalise_metric_key("") is None
    assert presets.pipeline_metric_key("volatility") == "Volatility"
    assert presets.pipeline_metric_key(None) is None
