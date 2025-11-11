"""Soft-coverage tests for :mod:`trend_analysis.presets`."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from trend_analysis import presets


@pytest.fixture()
def preset_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    base = tmp_path / "primary"
    extra = tmp_path / "extra"
    base.mkdir()
    extra.mkdir()

    monkeypatch.setattr(presets, "PRESETS_DIR", base)
    monkeypatch.setenv("TREND_PRESETS_DIR", str(extra))
    presets._preset_registry.cache_clear()
    yield base, extra
    presets._preset_registry.cache_clear()


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_candidate_dirs_include_environment_override(
    preset_workspace: tuple[Path, Path],
) -> None:
    base, extra = preset_workspace
    dirs = presets._candidate_preset_dirs()
    assert dirs == (base, extra)


def test_registry_honours_override_precedence(
    preset_workspace: tuple[Path, Path], caplog: pytest.LogCaptureFixture
) -> None:
    base, extra = preset_workspace
    caplog.set_level("WARNING")

    _write_yaml(
        base / "global.yml",
        """
name: Global Base
signals:
  window: 75
        """.strip(),
    )
    _write_yaml(
        extra / "global.yml",
        """
name: Global Override
signals:
  window: 55
        """.strip(),
    )

    registry = presets._preset_registry()
    preset = registry["global"]
    assert preset.label == "Global Override"
    assert preset.trend_spec.window == 55
    assert any(
        "Duplicate trend preset slug" in record.message for record in caplog.records
    )


def test_trend_preset_helpers_normalise_values() -> None:
    config = {
        "lookback_months": "48",
        "rebalance_frequency": "weekly",
        "selection_count": "5",
        "metrics": {"Sharpe": "2", "ignored": "x"},
        "signals": {
            "window": 40,
            "lag": 2,
            "vol_adjust": True,
            "vol_target": 0.25,
            "min_periods": 60,
        },
        "vol_adjust": {"window": {"length": None}},
    }

    spec = presets._build_trend_spec(config)
    preset = presets.TrendPreset(
        slug="balanced",
        label="Balanced",
        description="",
        trend_spec=spec,
        _config=presets._freeze_mapping(dict(config)),
    )

    defaults = preset.form_defaults()
    assert defaults["lookback_months"] == 48
    assert defaults["selection_count"] == 5
    assert defaults["metrics"] == {"sharpe": 2.0}

    mapping = preset.signals_mapping()
    assert mapping["window"] == 40
    assert mapping["min_periods"] == 40  # clamped to window

    vol_adjust = preset.vol_adjust_defaults()
    assert vol_adjust["enabled"] is True
    assert vol_adjust["target_vol"] == 0.25
    assert vol_adjust["window"]["length"] == 40

    pipeline_metrics = preset.metrics_pipeline()
    assert pipeline_metrics == {"Sharpe": 2.0}


def test_form_defaults_handles_missing_portfolio() -> None:
    preset = presets.TrendPreset(
        slug="defaults",
        label="Defaults",
        description="",
        trend_spec=presets.TrendSpec(),
        _config=presets._freeze_mapping({}),
    )
    defaults = preset.form_defaults()
    assert defaults["cooldown_months"] == 3
    assert defaults["risk_target"] == 0.1


def test_apply_trend_preset_updates_config() -> None:
    spec = presets.TrendSpec(window=30, lag=1, vol_adjust=False)
    preset = presets.TrendPreset(
        slug="core",
        label="Core",
        description="",
        trend_spec=spec,
        _config=presets._freeze_mapping({}),
    )

    config = SimpleNamespace(signals={"window": 10}, vol_adjust={}, run={})
    presets.apply_trend_preset(config, preset)

    assert config.signals["window"] == 30
    assert config.run["trend_preset"] == "core"
    assert config.vol_adjust["window"]["length"] == 30


def test_preset_lookup_and_alias_helpers(preset_workspace: tuple[Path, Path]) -> None:
    base, _ = preset_workspace
    _write_yaml(
        base / "momentum.yml",
        """
name: Momentum Alpha
signals:
  window: 45
metrics:
  sharpe: 1.5
        """.strip(),
    )

    presets._preset_registry.cache_clear()
    preset = presets.get_trend_preset("momentum")
    assert preset.slug == "momentum"
    assert presets.get_trend_preset("Momentum Alpha").slug == "momentum"

    slugs = presets.list_preset_slugs()
    assert "momentum" in slugs

    labels = [item.label for item in presets.list_trend_presets()]
    assert any(label == "Momentum Alpha" for label in labels)

    assert presets.normalise_metric_key("Sharpe_Ratio") == "sharpe"
    assert presets.pipeline_metric_key("drawdown") == "MaxDrawdown"


def test_metric_key_helpers_ignore_unknown_values() -> None:
    assert presets.normalise_metric_key("") is None
    assert presets.pipeline_metric_key("unknown") is None


def test_helper_functions_cover_edge_cases(tmp_path: Path) -> None:
    # Edge case 1: string coercion
    weights = presets._normalise_metric_weights({"Sharpe": "3"})
    assert weights == {"sharpe": 3.0}

    # Edge case 2: invalid object type should be ignored
    weights = presets._normalise_metric_weights({"invalid": object()})
    assert weights == {}

    # Edge case 3: empty string key should be ignored
    weights = presets._normalise_metric_weights({"": 1})
    assert weights == {}
    assert presets._normalise_metric_weights({"Sharpe": "bad"}) == {}

    assert presets._coerce_int("not-an-int", default=7, minimum=3) == 7
    assert presets._coerce_int(1, default=0, minimum=3) == 3

    assert presets._coerce_optional_int("9", minimum=5) == 9
    assert presets._coerce_optional_int("-1", minimum=5) is None
    assert presets._coerce_optional_int(None) is None
    assert presets._coerce_optional_int(object()) is None

    assert presets._coerce_optional_float("1.2", minimum=0.5) == 1.2
    assert presets._coerce_optional_float("0.1", minimum=0.5) is None
    assert presets._coerce_optional_float(object()) is None

    bogus_file = tmp_path / "invalid.yml"
    bogus_file.write_text("just-a-string", encoding="utf-8")
    bogus = presets._load_yaml(bogus_file)
    assert bogus == {}

    with pytest.raises(KeyError):
        presets.get_trend_preset("")


def test_build_trend_spec_handles_non_mapping_signals() -> None:
    spec = presets._build_trend_spec({"signals": ["not", "mapping"]})
    assert spec.window == 63
    assert spec.min_periods is None


def test_vol_adjust_defaults_uses_fallbacks() -> None:
    preset = presets.TrendPreset(
        slug="fallback",
        label="Fallback",
        description="",
        trend_spec=presets.TrendSpec(),
        _config=presets._freeze_mapping({}),
    )
    defaults = preset.vol_adjust_defaults()
    assert defaults["enabled"] is presets._DEFAULT_VOL_ADJUST["enabled"]
    assert defaults["target_vol"] == presets._DEFAULT_VOL_ADJUST["target_vol"]
    assert (
        defaults["window"]["length"] == presets._DEFAULT_VOL_ADJUST["window"]["length"]
    )


def test_vol_adjust_defaults_reads_signal_overrides() -> None:
    config = {"signals": {"vol_adjust": True, "window": 30}}
    spec = presets._build_trend_spec(config)
    preset = presets.TrendPreset(
        slug="signals",
        label="Signals",
        description="",
        trend_spec=spec,
        _config=presets._freeze_mapping(dict(config)),
    )
    defaults = preset.vol_adjust_defaults()
    assert defaults["enabled"] is True


def test_vol_adjust_defaults_handles_mapping_proxy() -> None:
    proxy_window = MappingProxyType({"length": None})
    config = {"vol_adjust": MappingProxyType({"window": proxy_window})}
    preset = presets.TrendPreset(
        slug="proxy",
        label="Proxy",
        description="",
        trend_spec=presets.TrendSpec(window=45, vol_adjust=False),
        _config=presets._freeze_mapping(dict(config)),
    )
    defaults = preset.vol_adjust_defaults()
    assert defaults["window"]["length"] == 45


def test_candidate_dirs_include_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "primary"
    base.mkdir()
    defaults_parent = tmp_path / "config" / "presets"
    defaults_parent.mkdir(parents=True)
    env = tmp_path / "env"
    env.mkdir()

    monkeypatch.setattr(presets, "PRESETS_DIR", base)
    monkeypatch.setattr(presets, "_DEFAULT_PRESETS_DIR", base)
    monkeypatch.setenv("TREND_PRESETS_DIR", str(env))
    presets._preset_registry.cache_clear()

    dirs = presets._candidate_preset_dirs()
    assert dirs[0] == base
    assert dirs[-1] == env
    assert len(dirs) >= 2

    monkeypatch.setenv("TREND_PRESETS_DIR", str(base))
    presets._preset_registry.cache_clear()
    duplicate_dirs = presets._candidate_preset_dirs()
    assert duplicate_dirs.count(base) == 1


def test_preset_registry_ignores_empty_files(
    preset_workspace: tuple[Path, Path],
) -> None:
    base, _ = preset_workspace
    _write_yaml(base / "empty.yml", "")
    presets._preset_registry.cache_clear()
    registry = presets._preset_registry()
    assert "empty" not in registry


def test_apply_trend_preset_handles_non_mapping_sections() -> None:
    preset = presets.TrendPreset(
        slug="alt",
        label="Alt",
        description="",
        trend_spec=presets.TrendSpec(window=20, lag=1, vol_adjust=True),
        _config=presets._freeze_mapping({}),
    )

    config = SimpleNamespace(
        signals=None, vol_adjust=MappingProxyType({"kept": 1}), run=None
    )
    presets.apply_trend_preset(config, preset)

    assert isinstance(config.signals, dict)
    assert config.vol_adjust["kept"] == 1
    assert config.run["trend_preset"] == "alt"


def test_apply_trend_preset_initialises_missing_sections() -> None:
    preset = presets.TrendPreset(
        slug="init",
        label="Init",
        description="",
        trend_spec=presets.TrendSpec(window=15, lag=2),
        _config=presets._freeze_mapping({}),
    )
    config = SimpleNamespace(signals="", vol_adjust=None, run=())
    presets.apply_trend_preset(config, preset)

    assert config.signals["window"] == 15
    assert config.vol_adjust["window"]["length"] == 15
    assert config.run["trend_preset"] == "init"
