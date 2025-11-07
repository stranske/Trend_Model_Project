from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from textwrap import dedent
from types import MappingProxyType

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


def test_normalise_metric_weights_skips_invalid_entries() -> None:
    weights = presets._normalise_metric_weights(
        {
            "sharpe_ratio": "1.5",
            "max_drawdown": object(),
            "unknown": 2,
        }
    )

    assert weights == {"sharpe": 1.5}


def test_form_defaults_ignores_non_mapping_portfolio() -> None:
    preset = presets.TrendPreset(
        slug="edge",
        label="Edge Case",
        description="",
        trend_spec=presets.TrendSpec(),
        _config={"portfolio": "none", "selection_count": 2},
    )

    defaults = preset.form_defaults()

    assert defaults["weighting_scheme"] == "equal"
    assert defaults["cooldown_months"] == 3
    assert defaults["selection_count"] == 2


def test_vol_adjust_defaults_uses_trend_spec_when_section_missing() -> None:
    preset = presets.TrendPreset(
        slug="trend-enabled",
        label="Trend Enabled",
        description="",
        trend_spec=presets.TrendSpec(window=45, vol_adjust=True, vol_target=0.4),
        _config={},
    )

    defaults = preset.vol_adjust_defaults()

    assert defaults["enabled"] is True
    assert defaults["target_vol"] == 0.4
    assert defaults["window"]["length"] == 45


def test_vol_adjust_defaults_prefers_config_section_over_signals() -> None:
    preset = presets.TrendPreset(
        slug="configured",
        label="Configured",
        description="",
        trend_spec=presets.TrendSpec(window=80, vol_adjust=False, vol_target=0.3),
        _config={
            "signals": {"vol_adjust": True},
            "vol_adjust": {
                "window": MappingProxyType({"length": None}),
                "target_vol": None,
            },
        },
    )

    defaults = preset.vol_adjust_defaults()

    assert defaults["enabled"] is False
    assert defaults["target_vol"] == 0.3
    assert defaults["window"] == {"length": 80}


def test_vol_adjust_defaults_copies_general_mapping() -> None:
    class CustomMapping(Mapping[str, int | None]):
        def __init__(self, data: dict[str, int | None]) -> None:
            self._data = data

        def __getitem__(self, key: str) -> int | None:
            return self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self) -> int:
            return len(self._data)

    window = CustomMapping({"length": None})
    preset = presets.TrendPreset(
        slug="recording",
        label="Recording",
        description="",
        trend_spec=presets.TrendSpec(),
        _config={"vol_adjust": {"window": window}},
    )

    defaults = preset.vol_adjust_defaults()

    assert isinstance(defaults["window"], dict)
    assert defaults["window"] == {"length": 55}
    assert defaults["window"] is not window


def test_apply_trend_preset_converts_mapping_sections() -> None:
    preset = presets.TrendPreset(
        slug="mapping",
        label="Mapping",
        description="",
        trend_spec=presets.TrendSpec(window=70, vol_adjust=True, vol_target=0.2),
        _config={"vol_adjust": {"window": {"length": 70}}},
    )

    class MappingConfig:
        def __init__(self) -> None:
            self.signals = MappingProxyType({"kind": "tsmom"})
            self.vol_adjust = MappingProxyType({"existing": 1})
            self.run = MappingProxyType({"previous": "value"})

    config = MappingConfig()
    presets.apply_trend_preset(config, preset)

    assert config.vol_adjust["existing"] == 1
    assert config.vol_adjust["enabled"] is True
    assert config.vol_adjust["window"]["length"] == 70
    assert config.run["previous"] == "value"
    assert config.run["trend_preset"] == "mapping"
