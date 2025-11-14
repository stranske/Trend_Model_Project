"""Focused coverage for trend presets helpers and registry."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from trend_analysis import presets
from trend_analysis.signals import TrendSpec


def test_normalise_metric_weights_skips_invalid_entries() -> None:
    weights = presets._normalise_metric_weights(
        {
            "Sharpe": "1.5",
            "return_ann": "bad",  # alias resolves but fails float conversion
            "": 1,
            "volatility": "0.5",
            "ignored": object(),
        }
    )

    assert weights == {"sharpe": 1.5, "vol": 0.5}


def test_coerce_helpers_enforce_bounds() -> None:
    assert presets._coerce_int("7", default=5, minimum=3) == 7
    assert presets._coerce_int(None, default=4, minimum=2) == 4
    assert presets._coerce_int(1, default=4, minimum=2) == 2

    assert presets._coerce_optional_int("4", minimum=2) == 4
    assert presets._coerce_optional_int("bad", minimum=2) is None
    assert presets._coerce_optional_int(1, minimum=2) is None

    assert presets._coerce_optional_float("0.5", minimum=0.1) == 0.5
    assert presets._coerce_optional_float("0.0", minimum=0.1) is None
    assert presets._coerce_optional_float("bad", minimum=0.1) is None


def test_build_trend_spec_clamps_optional_fields() -> None:
    spec = presets._build_trend_spec(
        {
            "signals": {
                "window": "20",
                "min_periods": 50,
                "lag": "2",
                "vol_adjust": "yes",
                "vol_target": "0.3",
                "zscore": True,
            }
        }
    )

    assert spec.window == 20
    assert spec.min_periods == 20  # min_periods cannot exceed the window
    assert spec.lag == 2
    assert spec.vol_adjust is True
    assert spec.vol_target == 0.3
    assert spec.zscore is True


def test_build_trend_spec_uses_defaults_when_signals_missing() -> None:
    spec = presets._build_trend_spec({})
    assert spec.window == 63
    assert spec.min_periods is None
    assert spec.lag == 1
    assert spec.vol_adjust is False
    assert spec.vol_target is None
    assert spec.zscore is False


@pytest.fixture()
def sample_preset_config() -> dict[str, Any]:
    return {
        "lookback_months": "24",
        "rebalance_frequency": "weekly",
        "min_track_months": "18",
        "selection_count": "5",
        "risk_target": "0.75",
        "metrics": {"Sharpe": "1.0", "vol": "2"},
        "portfolio": {"weighting_scheme": "equal", "cooldown_months": "1"},
        "signals": {
            "window": "45",
            "min_periods": "90",
            "lag": "3",
            "vol_adjust": False,
            "vol_target": "0.4",
        },
    }


@pytest.fixture()
def sample_preset(sample_preset_config: dict[str, Any]) -> presets.TrendPreset:
    spec = TrendSpec(
        window=45,
        min_periods=30,
        lag=3,
        vol_adjust=True,
        vol_target=0.4,
        zscore=True,
    )
    return presets.TrendPreset(
        slug="custom",
        label="Custom",
        description="Preset for coverage",
        trend_spec=spec,
        _config=presets._freeze_mapping(sample_preset_config),
    )


def test_trend_preset_form_defaults_and_metrics_pipeline(
    sample_preset: presets.TrendPreset,
) -> None:
    defaults = sample_preset.form_defaults()
    assert defaults["lookback_months"] == 24
    assert defaults["min_track_months"] == 18
    assert defaults["selection_count"] == 5
    assert defaults["risk_target"] == 0.75
    assert defaults["metrics"] == {"sharpe": 1.0, "vol": 2.0}

    pipeline_weights = sample_preset.metrics_pipeline()
    assert pipeline_weights == {"Sharpe": 1.0, "Volatility": 2.0}

    mapping = sample_preset.signals_mapping()
    assert mapping == {
        "kind": "tsmom",
        "window": 45,
        "lag": 3,
        "vol_adjust": True,
        "zscore": True,
        "min_periods": 30,
        "vol_target": 0.4,
    }


def test_trend_preset_form_defaults_handles_missing_portfolio(
    sample_preset_config: dict[str, Any],
) -> None:
    sample_preset_config.pop("portfolio", None)
    spec = TrendSpec(window=30, min_periods=None, lag=1)
    preset = presets.TrendPreset(
        slug="missing-portfolio",
        label="Missing portfolio",
        description="",
        trend_spec=spec,
        _config=presets._freeze_mapping(sample_preset_config),
    )

    defaults = preset.form_defaults()
    assert defaults["weighting_scheme"] == "equal"
    assert defaults["cooldown_months"] == 3


def test_vol_adjust_defaults_merge_sources(
    sample_preset_config: dict[str, Any],
) -> None:
    sample_preset_config["vol_adjust"] = {
        "enabled": None,
        "target_vol": None,
        "window": {"length": None},
    }
    spec = TrendSpec(
        window=55, min_periods=None, lag=1, vol_adjust=True, vol_target=0.6
    )
    preset = presets.TrendPreset(
        slug="vol",
        label="Vol",
        description="",
        trend_spec=spec,
        _config=presets._freeze_mapping(sample_preset_config),
    )

    vol_defaults = preset.vol_adjust_defaults()
    assert vol_defaults == {
        "enabled": True,
        "target_vol": 0.6,
        "window": {"length": 55},
    }


def test_vol_adjust_defaults_falls_back_to_signals_and_defaults(
    sample_preset_config: dict[str, Any],
) -> None:
    sample_preset_config.pop("vol_adjust", None)
    sample_preset_config.pop("signals", None)
    spec = TrendSpec(
        window=63, min_periods=None, lag=1, vol_adjust=False, vol_target=None
    )
    preset = presets.TrendPreset(
        slug="fallback",
        label="Fallback",
        description="",
        trend_spec=spec,
        _config=presets._freeze_mapping(sample_preset_config),
    )

    defaults = preset.vol_adjust_defaults()
    assert defaults == {
        "enabled": True,
        "target_vol": 0.3,
        "window": {"length": 55},
    }


def test_vol_adjust_defaults_uses_signal_override(
    sample_preset_config: dict[str, Any],
) -> None:
    sample_preset_config["signals"] = {"vol_adjust": True}
    sample_preset_config.pop("vol_adjust", None)
    spec = TrendSpec(
        window=63, min_periods=None, lag=1, vol_adjust=False, vol_target=None
    )
    preset = presets.TrendPreset(
        slug="signal-override",
        label="Signal override",
        description="",
        trend_spec=spec,
        _config=presets._freeze_mapping(sample_preset_config),
    )

    defaults = preset.vol_adjust_defaults()
    assert defaults["enabled"] is True


def test_vol_adjust_defaults_uses_spec_when_different(
    sample_preset_config: dict[str, Any],
) -> None:
    sample_preset_config.pop("vol_adjust", None)
    sample_preset_config.pop("signals", None)
    spec = TrendSpec(
        window=70, min_periods=None, lag=1, vol_adjust=True, vol_target=None
    )
    preset = presets.TrendPreset(
        slug="spec-override",
        label="Spec override",
        description="",
        trend_spec=spec,
        _config=presets._freeze_mapping(sample_preset_config),
    )

    defaults = preset.vol_adjust_defaults()
    assert defaults["enabled"] is True
    assert defaults["window"]["length"] == 70


def test_candidate_dirs_include_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    env_dir = tmp_path / "env"
    env_dir.mkdir()

    monkeypatch.setattr(presets, "PRESETS_DIR", base_dir)
    monkeypatch.delenv("TREND_PRESETS_DIR", raising=False)
    monkeypatch.setenv("TREND_PRESETS_DIR", str(env_dir))

    directories = presets._candidate_preset_dirs()
    assert directories == (base_dir, env_dir)


def test_candidate_dirs_include_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    original = presets.PRESETS_DIR
    try:
        monkeypatch.setattr(presets, "PRESETS_DIR", presets._DEFAULT_PRESETS_DIR)
        directories = presets._candidate_preset_dirs()
        assert presets._DEFAULT_PRESETS_DIR in directories
    finally:
        monkeypatch.setattr(presets, "PRESETS_DIR", original)


@pytest.fixture()
def preset_registry_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    base_dir = tmp_path / "presets_base"
    base_dir.mkdir()
    env_dir = tmp_path / "presets_env"
    env_dir.mkdir()

    (base_dir / "alpha.yml").write_text(
        """
name: Alpha
metrics:
  sharpe: 1
signals:
  window: 30
  lag: 1
  vol_adjust: false
        """.strip(),
        encoding="utf-8",
    )
    (base_dir / "beta.yml").write_text(
        """
name: Beta
metrics:
  vol: 2
signals:
  window: 40
  lag: 2
        """.strip(),
        encoding="utf-8",
    )

    (env_dir / "alpha.yml").write_text(
        """
name: Alpha Override
metrics:
  sharpe: 2
signals:
  window: 50
  lag: 1
        """.strip(),
        encoding="utf-8",
    )

    (env_dir / "invalid.yml").write_text("- not-a-mapping", encoding="utf-8")

    monkeypatch.setattr(presets, "PRESETS_DIR", base_dir)
    monkeypatch.delenv("TREND_PRESETS_DIR", raising=False)
    monkeypatch.setenv("TREND_PRESETS_DIR", str(env_dir))
    presets._preset_registry.cache_clear()
    yield base_dir, env_dir
    presets._preset_registry.cache_clear()


def test_preset_registry_warns_on_duplicate(
    preset_registry_paths, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("WARNING", logger=presets.LOGGER.name)
    registry = presets._preset_registry()

    assert registry["alpha"].label == "Alpha Override"
    assert registry["beta"].label == "Beta"
    assert "Duplicate trend preset slug" in caplog.text


def test_list_and_get_trend_presets(preset_registry_paths) -> None:
    presets._preset_registry.cache_clear()
    registry = presets._preset_registry()

    labels = [preset.label for preset in presets.list_trend_presets()]
    assert labels == sorted(labels, key=str.lower)

    assert presets.get_trend_preset("alpha").label == "Alpha Override"
    assert presets.get_trend_preset("Alpha Override").slug == "alpha"
    with pytest.raises(KeyError):
        presets.get_trend_preset("missing")
    with pytest.raises(KeyError):
        presets.get_trend_preset("")

    assert presets.list_preset_slugs() == tuple(sorted(registry.keys()))


def test_metric_key_helpers_handle_aliases() -> None:
    assert presets.normalise_metric_key("Sharpe_Ratio") == "sharpe"
    assert presets.normalise_metric_key("") is None
    assert presets.pipeline_metric_key("volatility") == "Volatility"
    assert presets.pipeline_metric_key("") is None
    assert presets.pipeline_metric_key("unknown") is None


def test_apply_trend_preset_updates_config(preset_registry_paths) -> None:
    presets._preset_registry.cache_clear()
    registry = presets._preset_registry()
    preset = registry["alpha"]

    config = SimpleNamespace(
        signals={"lag": 9},
        vol_adjust={"enabled": False},
        run={"other": "value"},
    )
    presets.apply_trend_preset(config, preset)

    assert config.signals["window"] == preset.trend_spec.window
    assert config.signals["vol_adjust"] is preset.trend_spec.vol_adjust
    assert config.vol_adjust["target_vol"] == preset.vol_adjust_defaults()["target_vol"]
    assert config.run["trend_preset"] == "alpha"


def test_apply_trend_preset_handles_non_mapping_sections(preset_registry_paths) -> None:
    presets._preset_registry.cache_clear()
    preset = presets._preset_registry()["alpha"]

    config = SimpleNamespace(signals=None, vol_adjust=None, run=None)
    presets.apply_trend_preset(config, preset)

    assert isinstance(config.signals, dict)
    assert isinstance(config.vol_adjust, dict)
    assert isinstance(config.run, dict)


def test_apply_trend_preset_wraps_mapping_proxies(preset_registry_paths) -> None:
    presets._preset_registry.cache_clear()
    preset = presets._preset_registry()["alpha"]

    config = SimpleNamespace(
        signals=MappingProxyType({"lag": 5}),
        vol_adjust=MappingProxyType({"window": {"length": 10}}),
        run=MappingProxyType({"other": "value"}),
    )
    presets.apply_trend_preset(config, preset)

    assert config.signals["lag"] == preset.trend_spec.lag
    assert isinstance(config.vol_adjust, dict)
    assert isinstance(config.run, dict)
