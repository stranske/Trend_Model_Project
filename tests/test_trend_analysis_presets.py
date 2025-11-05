from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from trend_analysis import presets as preset_module


@pytest.fixture(autouse=True)
def clear_preset_cache() -> None:
    preset_module._preset_registry.cache_clear()
    yield
    preset_module._preset_registry.cache_clear()


def test_freeze_mapping_returns_immutable_view() -> None:
    frozen = preset_module._freeze_mapping({"a": 1, "b": 2})
    assert frozen["a"] == 1
    with pytest.raises(TypeError):
        frozen["c"] = 3  # type: ignore[misc]


def test_normalise_metric_weights_discards_invalid_entries() -> None:
    weights = preset_module._normalise_metric_weights(
        {"sharpe_ratio": "2", "bad": "x", "drawdown": 0.5}
    )
    assert weights == {"sharpe": 2.0, "drawdown": 0.5}


def test_normalise_metric_weights_handles_missing_aliases() -> None:
    weights = preset_module._normalise_metric_weights({"": 5, "none": None})
    assert weights == {}


def test_coerce_helpers_normalise_inputs() -> None:
    assert preset_module._coerce_int("7", default=3) == 7
    assert preset_module._coerce_int(None, default=5) == 5
    assert preset_module._coerce_optional_int("9") == 9
    assert preset_module._coerce_optional_int(-1) is None
    assert preset_module._coerce_optional_float("0.5") == 0.5
    assert preset_module._coerce_optional_float(-0.1) is None


def test_coerce_optional_helpers_return_none_for_invalid_values() -> None:
    assert preset_module._coerce_optional_int("abc") is None
    assert preset_module._coerce_optional_int(0, minimum=2) is None
    assert preset_module._coerce_optional_float("abc") is None
    assert preset_module._coerce_optional_float(-0.5) is None


def test_build_trend_spec_clamps_min_periods() -> None:
    spec = preset_module._build_trend_spec(
        {
            "signals": {
                "window": 20,
                "min_periods": 40,
                "lag": "3",
                "vol_adjust": True,
                "vol_target": "0.3",
                "zscore": True,
            }
        }
    )
    assert spec.window == 20
    assert spec.min_periods == 20
    assert spec.lag == 3
    assert spec.vol_adjust is True
    assert spec.vol_target == pytest.approx(0.3)
    assert spec.zscore is True


def test_build_trend_spec_handles_missing_signals() -> None:
    spec = preset_module._build_trend_spec({"signals": [1, 2, 3]})
    assert spec.window == 63
    assert spec.min_periods is None
    assert spec.vol_adjust is False


def test_trend_preset_helpers_expose_expected_defaults() -> None:
    spec = preset_module._build_trend_spec(
        {
            "signals": {
                "window": 50,
                "min_periods": 30,
                "lag": 2,
                "vol_adjust": True,
                "vol_target": 0.15,
                "zscore": False,
            }
        }
    )
    preset = preset_module.TrendPreset(
        slug="alpha",
        label="Alpha",
        description="demo",
        trend_spec=spec,
        _config=preset_module._freeze_mapping(
            {
                "lookback_months": 18,
                "min_track_months": 6,
                "selection_count": 7,
                "risk_target": 0.2,
                "rebalance_frequency": "weekly",
                "metrics": {"sharpe": "1.5", "drawdown": 0.4},
                "portfolio": {"weighting_scheme": "equal", "cooldown_months": 2},
                "vol_adjust": {"enabled": False, "window": {"slow": 63}},
            }
        ),
    )

    defaults = preset.form_defaults()
    assert defaults == {
        "lookback_months": 18,
        "rebalance_frequency": "weekly",
        "min_track_months": 6,
        "selection_count": 7,
        "risk_target": 0.2,
        "weighting_scheme": "equal",
        "cooldown_months": 2,
        "metrics": {"sharpe": 1.5, "drawdown": 0.4},
    }

    mapping = preset.signals_mapping()
    assert mapping == {
        "kind": "tsmom",
        "window": 50,
        "lag": 2,
        "vol_adjust": True,
        "zscore": False,
        "min_periods": 30,
        "vol_target": 0.15,
    }

    vol_adjust_defaults = preset.vol_adjust_defaults()
    assert vol_adjust_defaults["enabled"] is False
    assert vol_adjust_defaults["target_vol"] == pytest.approx(0.15)
    assert vol_adjust_defaults["window"]["slow"] == 63
    assert vol_adjust_defaults["window"]["length"] == 50
    assert vol_adjust_defaults is not preset._config.get("vol_adjust")

    pipeline_metrics = preset.metrics_pipeline()
    assert pipeline_metrics == {"Sharpe": 1.5, "MaxDrawdown": 0.4}


def test_trend_preset_defaults_use_fallbacks_when_config_missing() -> None:
    spec = preset_module._build_trend_spec({})
    preset = preset_module.TrendPreset(
        slug="base",
        label="Base",
        description="",
        trend_spec=spec,
        _config=_freeze_mapping({}),
    )

    defaults = preset.vol_adjust_defaults()
    assert defaults["enabled"] is True
    assert defaults["target_vol"] == 0.3
    assert defaults["window"] == {"length": 55}


def test_vol_adjust_defaults_respects_explicit_enabled_flag() -> None:
    spec = TrendSpec(
        window=21,
        min_periods=None,
        lag=1,
        vol_adjust=True,
        vol_target=0.6,
        zscore=True,
    )
    preset = TrendPreset(
        slug="cautious",
        label="Cautious",
        description="",
        trend_spec=spec,
        _config=_freeze_mapping(
            {
                "vol_adjust": {
                    "enabled": False,
                    "window": MappingProxyType({"length": 12}),
                    "target_vol": 0.4,
                }
            }
        ),
    )

    defaults = preset.vol_adjust_defaults()
    assert defaults["enabled"] is False
    assert defaults["target_vol"] == 0.4
    assert defaults["window"] == {"length": 12}


def test_apply_trend_preset_merges_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_registry(tmp_path, monkeypatch)
    preset = get_trend_preset("zulu")

    config = SimpleNamespace(
        signals={"existing": "keep"},
        vol_adjust={"enabled": False},
        run={"other": "value"},
    )

    defaults = preset.form_defaults()
    assert defaults["weighting_scheme"] == "equal"
    assert defaults["cooldown_months"] == 3

    vol_defaults = preset.vol_adjust_defaults()
    assert vol_defaults["enabled"] is False
    assert vol_defaults["window"]["length"] == spec.window


def test_candidate_preset_dirs_prefers_base_then_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = tmp_path / "base"
    env_dir = tmp_path / "env"
    base.mkdir()
    env_dir.mkdir()

    monkeypatch.setattr(preset_module, "PRESETS_DIR", base)
    monkeypatch.setenv("TREND_PRESETS_DIR", str(env_dir))

    candidates = preset_module._candidate_preset_dirs()
    assert candidates == (base, env_dir)


def test_candidate_preset_dirs_includes_repository_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preset_module, "PRESETS_DIR", preset_module._DEFAULT_PRESETS_DIR)
    monkeypatch.delenv("TREND_PRESETS_DIR", raising=False)
    candidates = preset_module._candidate_preset_dirs()
    assert preset_module._DEFAULT_PRESETS_DIR in candidates
    assert isinstance(candidates, tuple)


def test_preset_registry_loads_yaml_and_handles_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    base = tmp_path / "base"
    env_dir = tmp_path / "env"
    base.mkdir()
    env_dir.mkdir()

    (base / "alpha.yml").write_text(
        """
name: Alpha Base
signals:
  window: 40
  min_periods: 20
  lag: 1
"""
    )

    (env_dir / "alpha.yml").write_text(
        """
name: Alpha Override
description: Override description
signals:
  window: 84
  min_periods: 63
  lag: 1
  vol_adjust: true
  vol_target: 0.1
  zscore: true
lookback_months: 18
min_track_months: 6
selection_count: 8
risk_target: 0.25
rebalance_frequency: monthly
metrics:
  sharpe: "1.5"
  max_drawdown: "0.3"
portfolio:
  weighting_scheme: equal
  cooldown_months: 2
vol_adjust:
  enabled: false
  window:
    fast: 21
"""
    )

    (env_dir / "beta.yml").write_text(
        """
name: Beta Preset
signals:
  window: 30
  min_periods: 15
  lag: 2
"""
    )

    monkeypatch.setattr(preset_module, "PRESETS_DIR", base)
    monkeypatch.setenv("TREND_PRESETS_DIR", str(env_dir))
    caplog.set_level("WARNING", "trend_analysis.presets")

    registry = preset_module._preset_registry()
    assert set(registry.keys()) == {"alpha", "beta"}
    assert "overrides definition" in caplog.text

    alpha = registry["alpha"]
    defaults = alpha.form_defaults()
    assert defaults["metrics"] == {"sharpe": 1.5, "drawdown": 0.3}
    assert defaults["risk_target"] == 0.25

    assert preset_module.list_preset_slugs() == ("alpha", "beta")
    assert [p.slug for p in preset_module.list_trend_presets()] == ["alpha", "beta"]

    assert preset_module.get_trend_preset("beta").label == "Beta Preset"
    assert preset_module.get_trend_preset("Alpha Override").slug == "alpha"
    with pytest.raises(KeyError):
        preset_module.get_trend_preset("")

    config = SimpleNamespace(signals={"lag": 9}, vol_adjust={}, run={})
    preset_module.apply_trend_preset(config, alpha)
    assert config.signals["window"] == 84
    assert config.signals["vol_adjust"] is True
    assert config.vol_adjust["enabled"] is False
    assert config.vol_adjust["target_vol"] == pytest.approx(0.1)
    assert config.run["trend_preset"] == "alpha"


def test_preset_registry_skips_non_mapping_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "empty.yml").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(preset_module, "PRESETS_DIR", base)
    monkeypatch.setattr(preset_module, "_DEFAULT_PRESETS_DIR", base)
    monkeypatch.delenv("TREND_PRESETS_DIR", raising=False)
    monkeypatch.setattr(preset_module, "_candidate_preset_dirs", lambda: (base,))

    registry = preset_module._preset_registry()
    assert registry == {}


def test_metric_alias_helpers_cover_known_aliases() -> None:
    assert preset_module.normalise_metric_key("Sharpe_Ratio") == "sharpe"
    assert preset_module.normalise_metric_key("") is None
    assert preset_module.pipeline_metric_key("max_drawdown") == "MaxDrawdown"
    assert preset_module.pipeline_metric_key(None) is None  # type: ignore[arg-type]


def test_load_yaml_returns_empty_mapping_for_invalid_payload(tmp_path: Path) -> None:
    yaml_path = tmp_path / "invalid.yml"
    yaml_path.write_text("[]", encoding="utf-8")
    assert preset_module._load_yaml(yaml_path) == {}


def test_get_trend_preset_raises_for_unknown_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preset_module, "PRESETS_DIR", tmp_path := Path("nonexistent"))
    preset_module._preset_registry.cache_clear()
    with pytest.raises(KeyError):
        preset_module.get_trend_preset("missing")


def test_apply_trend_preset_handles_non_mapping_sections(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = tmp_path / "base"
    env_dir = tmp_path / "env"
    base.mkdir()
    env_dir.mkdir()

    (env_dir / "alpha.yml").write_text(
        """
name: Alpha Override
signals:
  window: 20
  lag: 1
  vol_adjust: false
"""
    )

    monkeypatch.setattr(preset_module, "PRESETS_DIR", base)
    monkeypatch.setattr(preset_module, "_DEFAULT_PRESETS_DIR", base)
    monkeypatch.setenv("TREND_PRESETS_DIR", str(env_dir))

    preset = preset_module.get_trend_preset("alpha override")

    class Dummy:
        signals = ["not", "mapping"]
        vol_adjust = ()
        run = ()

    config = Dummy()
    preset_module.apply_trend_preset(config, preset)
    assert isinstance(config.signals, dict)
    assert isinstance(config.vol_adjust, dict)
    assert isinstance(config.run, dict)
