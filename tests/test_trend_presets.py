from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import pytest
import yaml

from trend_analysis import presets
from trend_analysis.signals import TrendSpec


@dataclass
class DummyConfig:
    signals: Any = MappingProxyType({})
    vol_adjust: Any = MappingProxyType({})
    run: Any = MappingProxyType({})


def _write_yaml(path: Any, data: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.fixture
def preset_environment(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Any]:
    base_dir = tmp_path / "base"
    env_dir = tmp_path / "env"
    base_dir.mkdir()
    env_dir.mkdir()

    monkeypatch.setattr(presets, "PRESETS_DIR", base_dir)
    monkeypatch.setenv("TREND_PRESETS_DIR", str(env_dir))
    presets._preset_registry.cache_clear()

    yield base_dir, env_dir

    presets._preset_registry.cache_clear()
    monkeypatch.delenv("TREND_PRESETS_DIR", raising=False)


def test_preset_registry_honours_precedence(
    preset_environment: tuple[Any, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    base_dir, env_dir = preset_environment

    (base_dir / "ignored.yml").write_text("- not a mapping\n", encoding="utf-8")

    _write_yaml(
        base_dir / "alpha.yml",
        {
            "name": "Alpha Base",
            "description": "Base preset",
            "lookback_months": 12,
            "selection_count": 5,
            "metrics": {"Sharpe": 1, "vol": 0.5, "skip": "n/a"},
            "signals": {"window": 50, "min_periods": 60, "lag": 2, "vol_adjust": True},
        },
    )

    _write_yaml(
        env_dir / "alpha.yml",
        {
            "name": "Alpha Override",
            "description": "Override preset",
            "lookback_months": 18,
            "selection_count": 7,
            "signals": {"window": 42, "lag": 3, "vol_adjust": False, "vol_target": 0.2},
            "vol_adjust": {
                "enabled": False,
                "target_vol": 0.15,
                "window": {"length": 30},
            },
        },
    )

    _write_yaml(
        env_dir / "beta.yml",
        {
            "name": "Beta",
            "description": "Second preset",
            "metrics": {"sharpe_ratio": "1.0", "max_drawdown": 0.25},
            "signals": {"window": 63, "lag": 1, "vol_adjust": True, "vol_target": 0.1},
        },
    )

    with caplog.at_level("WARNING"):
        registry = presets._preset_registry()

    assert "Duplicate trend preset slug" in " ".join(caplog.messages)
    assert set(registry.keys()) == {"alpha", "beta"}

    alpha = presets.get_trend_preset("alpha")
    assert alpha.label == "Alpha Override"
    assert alpha.slug == "alpha"
    assert alpha.trend_spec.window == 42
    assert alpha.trend_spec.vol_target == 0.2
    assert alpha.trend_spec.vol_adjust is False

    assert presets.get_trend_preset("Alpha Override") is alpha

    beta = presets.get_trend_preset("Beta")
    assert beta.metrics_pipeline()["Sharpe"] == 1.0
    assert beta.metrics_pipeline()["MaxDrawdown"] == 0.25

    assert presets.list_preset_slugs() == ("alpha", "beta")
    labels = tuple(preset.label for preset in presets.list_trend_presets())
    assert labels == ("Alpha Override", "Beta")


def test_get_trend_preset_supports_label_lookup(
    preset_environment: tuple[Any, Any],
) -> None:
    base_dir, _ = preset_environment

    _write_yaml(
        base_dir / "gamma.yml",
        {
            "name": "Gamma Label",
            "signals": {"window": 84, "lag": 2},
        },
    )

    preset = presets.get_trend_preset("Gamma Label")
    assert preset.label == "Gamma Label"

    with pytest.raises(KeyError, match="Preset name must be provided"):
        presets.get_trend_preset("")


def test_get_trend_preset_raises_for_unknown(
    preset_environment: tuple[Any, Any],
) -> None:
    base_dir, _ = preset_environment
    _write_yaml(
        base_dir / "zeta.yml",
        {
            "name": "Zeta",
            "signals": {"window": 30, "lag": 1},
        },
    )
    presets.get_trend_preset("Zeta")
    with pytest.raises(KeyError, match="Unknown trend preset"):
        presets.get_trend_preset("Missing")


def test_apply_trend_preset_merges_config(preset_environment: tuple[Any, Any]) -> None:
    base_dir, _ = preset_environment

    _write_yaml(
        base_dir / "gamma.yml",
        {
            "name": "Gamma",
            "signals": {"window": 84, "lag": 2, "vol_adjust": True, "vol_target": 0.1},
            "vol_adjust": {"enabled": True, "target_vol": 0.12},
        },
    )

    preset = presets.get_trend_preset("gamma")
    config = DummyConfig(
        signals={"existing": True}, vol_adjust={"enabled": False}, run={}
    )

    presets.apply_trend_preset(config, preset)

    assert config.signals["kind"] == "tsmom"
    assert config.signals["window"] == 84
    assert config.signals["lag"] == 2
    assert config.signals["vol_adjust"] is True

    assert config.vol_adjust["enabled"] is True
    assert config.vol_adjust["target_vol"] == 0.12
    assert config.vol_adjust["window"]["length"] == 84

    assert config.run["trend_preset"] == "gamma"


def test_apply_trend_preset_handles_non_mapping_attributes(
    preset_environment: tuple[Any, Any],
) -> None:
    base_dir, _ = preset_environment

    _write_yaml(
        base_dir / "epsilon.yml",
        {
            "name": "Epsilon",
            "signals": {"window": 21, "lag": 1},
        },
    )

    preset = presets.get_trend_preset("epsilon")
    config = DummyConfig(signals=None, vol_adjust=(), run=None)

    presets.apply_trend_preset(config, preset)

    assert isinstance(config.signals, dict)
    assert isinstance(config.vol_adjust, dict)
    assert isinstance(config.run, dict)
    assert config.run["trend_preset"] == "epsilon"


def test_apply_trend_preset_handles_mapping_proxy_attributes(
    preset_environment: tuple[Any, Any],
) -> None:
    base_dir, _ = preset_environment

    _write_yaml(
        base_dir / "eta.yml",
        {
            "name": "Eta",
            "signals": {"window": 40, "lag": 1},
        },
    )

    preset = presets.get_trend_preset("eta")
    config = DummyConfig(
        signals=MappingProxyType({"legacy": True}),
        vol_adjust=MappingProxyType({"enabled": False}),
        run=MappingProxyType({"previous": "value"}),
    )

    presets.apply_trend_preset(config, preset)

    assert "enabled" in config.vol_adjust
    assert config.run["trend_preset"] == "eta"


def test_helper_functions_cover_edge_cases() -> None:
    weights = presets._normalise_metric_weights(
        {"Sharpe_Ratio": "2", "sharpe": "bad", "unknown": 1}
    )
    assert weights == {"sharpe": 2.0}

    assert presets._normalise_metric_weights({"sharpe": object()}) == {}

    assert presets._coerce_int("3", default=1, minimum=2) == 3
    assert presets._coerce_int("bad", default=5, minimum=1) == 5

    assert presets._coerce_optional_int("4", minimum=2) == 4
    assert presets._coerce_optional_int("1", minimum=2) is None
    assert presets._coerce_optional_int("bad", minimum=1) is None
    assert presets._coerce_optional_int(None) is None

    assert presets._coerce_optional_float("0.5", minimum=0.2) == 0.5
    assert presets._coerce_optional_float("0.1", minimum=0.2) is None
    assert presets._coerce_optional_float({"oops": 1}, minimum=0.0) is None

    assert presets.normalise_metric_key("ShArPe_ratio") == "sharpe"
    assert presets.normalise_metric_key("") is None
    assert presets.normalise_metric_key("unknown") is None

    assert presets.pipeline_metric_key("volatility") == "Volatility"
    assert presets.pipeline_metric_key("") is None


def test_build_trend_spec_handles_missing_signals() -> None:
    default_spec = presets._build_trend_spec({})
    assert default_spec.window == 63
    assert default_spec.lag == 1
    assert default_spec.min_periods is None

    overridden = presets._build_trend_spec(
        {
            "signals": {
                "window": 10,
                "min_periods": 20,
                "lag": 0,
                "vol_adjust": 1,
                "vol_target": "0.5",
                "zscore": True,
            }
        }
    )
    assert overridden.window == 10
    assert overridden.min_periods == 10
    assert overridden.lag == 1
    assert overridden.vol_adjust is True
    assert overridden.vol_target == 0.5
    assert overridden.zscore is True


def test_trend_preset_defaults_with_sparse_config() -> None:
    spec = TrendSpec(
        window=45, min_periods=30, lag=3, vol_adjust=True, vol_target=None, zscore=True
    )
    raw_config = MappingProxyType(
        {
            "portfolio": None,
            "metrics": ["not", "mapping"],
            "signals": {"vol_adjust": True},
        }
    )
    preset = presets.TrendPreset(
        slug="custom",
        label="Custom",
        description="",
        trend_spec=spec,
        _config=raw_config,,
    )

    defaults = preset.form_defaults()
    assert defaults["weighting_scheme"] == "equal"
    assert defaults["metrics"] == {}
    assert defaults["risk_target"] == 0.1

    overrides = preset.vol_adjust_defaults()
    assert overrides["enabled"] is True
    assert overrides["target_vol"] == presets._DEFAULT_VOL_ADJUST["target_vol"]
    assert overrides["window"]["length"] == 45


def test_signals_mapping_includes_optional_fields() -> None:
    spec = TrendSpec(
        window=25, min_periods=10, lag=2, vol_adjust=True, vol_target=0.3, zscore=False
    )
    preset = presets.TrendPreset(
        slug="sig",
        label="Signals",
        description="",
        trend_spec=spec,
        _config=MappingProxyType({}),
    )
    mapping = preset.signals_mapping()
    assert mapping["min_periods"] == 10
    assert mapping["vol_target"] == 0.3


def test_vol_adjust_defaults_prefers_spec_when_section_present() -> None:
    spec = TrendSpec(
        window=80,
        min_periods=None,
        lag=1,
        vol_adjust=True,
        vol_target=0.4,
        zscore=False,
    )
    raw_config = MappingProxyType({"vol_adjust": {"window": MappingProxyType({})}})
    preset = presets.TrendPreset(
        slug="override",
        label="Override",
        description="",
        trend_spec=spec,
        _config=raw_config,
    )

    overrides = preset.vol_adjust_defaults()
    assert overrides["enabled"] is True
    assert overrides["target_vol"] == 0.4
    assert overrides["window"]["length"] == 80


def test_vol_adjust_defaults_uses_spec_when_no_overrides() -> None:
    spec = TrendSpec(
        window=90,
        min_periods=None,
        lag=1,
        vol_adjust=True,
        vol_target=None,
        zscore=False,
    )
    preset = presets.TrendPreset(
        slug="no-override",
        label="No Override",
        description="",
        trend_spec=spec,
        _config=MappingProxyType({}),
    )
    overrides = preset.vol_adjust_defaults()
    assert overrides["enabled"] is True


def test_vol_adjust_defaults_handles_readonly_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = TrendSpec(
        window=70,
        min_periods=None,
        lag=1,
        vol_adjust=True,
        vol_target=None,
        zscore=False,
    )
    raw_config = MappingProxyType(
        {"vol_adjust": {"window": MappingProxyType({"length": None})}}
    )
    preset = presets.TrendPreset(
        slug="readonly",
        label="Readonly",
        description="",
        trend_spec=spec,
        _config=raw_config,
    )
    with monkeypatch.context() as patcher:
        patcher.setattr(presets, "MutableMapping", tuple)
        overrides = preset.vol_adjust_defaults()
    assert overrides["window"]["length"] == 70


def test_vol_adjust_defaults_respects_existing_configuration(
    preset_environment: tuple[Any, Any],
) -> None:
    base_dir, _ = preset_environment

    _write_yaml(
        base_dir / "delta.yml",
        {
            "name": "Delta",
            "signals": {"window": 63, "lag": 1, "vol_adjust": False},
            "vol_adjust": {"enabled": False},
        },
    )

    preset = presets.get_trend_preset("delta")

    overrides = preset.vol_adjust_defaults()
    assert overrides["enabled"] is False
    assert overrides["target_vol"] == presets._DEFAULT_VOL_ADJUST["target_vol"]
    assert (
        overrides["window"]["length"] == presets._DEFAULT_VOL_ADJUST["window"]["length"]
    )


def test_candidate_preset_dirs_deduplicates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    base_dir = tmp_path / "presets"
    base_dir.mkdir()

    monkeypatch.setattr(presets, "PRESETS_DIR", base_dir)
    monkeypatch.setenv("TREND_PRESETS_DIR", str(base_dir))
    presets._preset_registry.cache_clear()

    try:
        candidates = presets._candidate_preset_dirs()
    finally:
        presets._preset_registry.cache_clear()
        monkeypatch.delenv("TREND_PRESETS_DIR", raising=False)

    resolved = base_dir.resolve()
    assert list(candidates).count(resolved) == 1


def test_candidate_preset_dirs_includes_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setattr(presets, "PRESETS_DIR", presets._DEFAULT_PRESETS_DIR)
    extra = tmp_path / "extra"
    extra.mkdir()
    monkeypatch.setenv("TREND_PRESETS_DIR", str(extra))
    presets._preset_registry.cache_clear()

    try:
        dirs = presets._candidate_preset_dirs()
    finally:
        presets._preset_registry.cache_clear()
        monkeypatch.delenv("TREND_PRESETS_DIR", raising=False)

    assert presets._DEFAULT_PRESETS_DIR in dirs
    assert extra.resolve() in dirs


def test_load_yaml_returns_empty_for_non_mapping(tmp_path: Any) -> None:
    path = tmp_path / "invalid.yml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    assert presets._load_yaml(path) == {}
