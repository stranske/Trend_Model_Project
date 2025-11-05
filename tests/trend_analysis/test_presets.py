from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

import trend_analysis.presets as presets
from trend_analysis.signals import TrendSpec


@pytest.fixture(autouse=True)
def reset_registry(monkeypatch: pytest.MonkeyPatch):
    presets._preset_registry.cache_clear()
    monkeypatch.delenv("TREND_PRESETS_DIR", raising=False)
    yield
    presets._preset_registry.cache_clear()
    monkeypatch.delenv("TREND_PRESETS_DIR", raising=False)
    presets.PRESETS_DIR = presets._DEFAULT_PRESETS_DIR


@pytest.fixture()
def sample_config() -> dict[str, Any]:
    return {
        "lookback_months": "48",
        "rebalance_frequency": "Weekly",
        "min_track_months": "12",
        "selection_count": "25",
        "risk_target": "0.33",
        "metrics": {"Sharpe_Ratio": "2", "invalid": "abc"},
        "portfolio": {"cooldown_months": "6", "weighting_scheme": "risk"},
        "vol_adjust": {"window": {"length": 21}},
        "signals": {
            "window": "126",
            "min_periods": "200",
            "lag": "2",
            "vol_target": "0.7",
        },
    }


def test_freeze_mapping_returns_immutable_copy(sample_config: dict[str, Any]):
    frozen = presets._freeze_mapping(sample_config)
    assert dict(frozen) == sample_config
    with pytest.raises(TypeError):
        frozen["new"] = 1  # type: ignore[index]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"Sharpe": "1.5", "Return_Ann": "2"}, {"sharpe": 1.5, "return_ann": 2.0}),
        ({"Unknown": "abc", "": None}, {}),
        (
            {"Sharpe": "nan", "Max_Drawdown": 3},
            {"sharpe": float("nan"), "drawdown": 3.0},
        ),
    ],
)
def test_normalise_metric_weights_handles_aliases(
    raw: dict[str, Any], expected: dict[str, float]
):
    weights = presets._normalise_metric_weights(raw)
    assert weights.keys() == expected.keys()
    for key in expected:
        if key == "sharpe" and expected[key] != expected[key]:  # NaN comparison
            assert weights[key] != weights[key]
        else:
            assert weights[key] == pytest.approx(expected[key])


def test_normalise_metric_weights_skips_uncoercible_values():
    weights = presets._normalise_metric_weights({"sharpe": object()})
    assert weights == {}


def test_integer_coercion_helpers_apply_bounds():
    assert presets._coerce_int("10", default=5, minimum=1) == 10
    assert presets._coerce_int("bad", default=5, minimum=4) == 5
    assert presets._coerce_int(0, default=1, minimum=3) == 3
    assert presets._coerce_optional_int("12", minimum=5) == 12
    assert presets._coerce_optional_int("2", minimum=5) is None
    assert presets._coerce_optional_int(None) is None
    assert presets._coerce_optional_int("bad") is None
    assert presets._coerce_optional_float("0.7", minimum=0.5) == pytest.approx(0.7)
    assert presets._coerce_optional_float("0.2", minimum=0.5) is None
    assert presets._coerce_optional_float(None) is None
    assert presets._coerce_optional_float("bad") is None


def test_build_trend_spec_clamps_invalid_min_periods(sample_config: dict[str, Any]):
    spec = presets._build_trend_spec(sample_config)
    assert isinstance(spec, TrendSpec)
    assert spec.window == 126
    assert spec.min_periods == 126  # clamped to window
    assert spec.lag == 2
    assert spec.vol_adjust is False
    assert spec.vol_target == pytest.approx(0.7)


def test_build_trend_spec_handles_missing_signals():
    spec = presets._build_trend_spec({"signals": [1, 2, 3]})
    assert spec.window == 63
    assert spec.min_periods is None
    assert spec.vol_target is None


@pytest.fixture()
def sample_preset(sample_config: dict[str, Any]) -> presets.TrendPreset:
    spec = presets._build_trend_spec(sample_config)
    return presets.TrendPreset(
        slug="momentum",
        label="Momentum",
        description="Long-term trend following",
        trend_spec=spec,
        _config=presets._freeze_mapping(sample_config),
    )


def test_trend_preset_form_defaults_normalises_values(
    sample_preset: presets.TrendPreset,
):
    defaults = sample_preset.form_defaults()
    assert defaults["lookback_months"] == 48
    assert defaults["rebalance_frequency"] == "Weekly"
    assert defaults["selection_count"] == 25
    assert defaults["risk_target"] == pytest.approx(0.33)
    assert defaults["metrics"] == {"sharpe": 2.0}
    assert defaults["cooldown_months"] == 6


def test_trend_preset_form_defaults_handles_missing_portfolio(
    sample_config: dict[str, Any],
):
    custom = dict(sample_config)
    custom["portfolio"] = "not-a-mapping"
    preset = presets.TrendPreset(
        slug="defaults",
        label="Defaults",
        description="",
        trend_spec=presets._build_trend_spec(custom),
        _config=presets._freeze_mapping(custom),
    )
    defaults = preset.form_defaults()
    assert defaults["weighting_scheme"] == "equal"
    assert defaults["cooldown_months"] == 3


def test_trend_preset_signals_and_vol_adjust_defaults(
    sample_preset: presets.TrendPreset,
):
    mapping = sample_preset.signals_mapping()
    assert mapping["window"] == 126
    assert mapping["lag"] == 2
    defaults = sample_preset.vol_adjust_defaults()
    assert defaults["enabled"] is False
    assert defaults["target_vol"] == pytest.approx(0.7)
    assert defaults["window"]["length"] == 21


def test_metrics_pipeline_translates_keys(sample_preset: presets.TrendPreset):
    pipeline_weights = sample_preset.metrics_pipeline()
    assert pipeline_weights == {"Sharpe": 2.0}


def test_vol_adjust_defaults_preserves_existing_target(sample_config: dict[str, Any]):
    custom = dict(sample_config)
    custom["vol_adjust"] = {"target_vol": 0.2}
    preset = presets.TrendPreset(
        slug="custom",
        label="Custom",
        description="",
        trend_spec=presets._build_trend_spec(custom),
        _config=presets._freeze_mapping(custom),
    )
    defaults = preset.vol_adjust_defaults()
    assert defaults["target_vol"] == pytest.approx(0.2)


def test_vol_adjust_defaults_respects_existing_enabled_flag(
    sample_config: dict[str, Any],
):
    from types import MappingProxyType

    custom = dict(sample_config)
    custom["vol_adjust"] = {"enabled": True, "window": MappingProxyType({"length": 5})}
    preset = presets.TrendPreset(
        slug="flags",
        label="Flags",
        description="",
        trend_spec=presets._build_trend_spec(custom),
        _config=presets._freeze_mapping(custom),
    )
    defaults = preset.vol_adjust_defaults()
    assert defaults["enabled"] is True
    assert defaults["window"]["length"] == 5
    assert defaults["target_vol"] == pytest.approx(0.7)


def test_vol_adjust_defaults_handles_non_mapping_source(sample_config: dict[str, Any]):
    custom = dict(sample_config)
    custom["vol_adjust"] = "disabled"
    preset = presets.TrendPreset(
        slug="vol",
        label="Vol",
        description="",
        trend_spec=presets._build_trend_spec(custom),
        _config=presets._freeze_mapping(custom),
    )
    defaults = preset.vol_adjust_defaults()
    assert defaults["window"]["length"] == preset.trend_spec.window


def test_signals_mapping_omits_optional_fields_when_none():
    spec = presets._build_trend_spec({"signals": {}})
    preset = presets.TrendPreset(
        slug="base",
        label="Base",
        description="",
        trend_spec=spec,
        _config=presets._freeze_mapping({"signals": {}}),
    )
    mapping = preset.signals_mapping()
    assert "min_periods" not in mapping
    assert "vol_target" not in mapping


def test_load_yaml_and_candidate_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    overrides = tmp_path / "overrides"
    overrides.mkdir()
    empty_yaml = overrides / "empty.yml"
    empty_yaml.write_text("[]", encoding="utf-8")
    assert presets._load_yaml(empty_yaml) == {}

    env_dir = tmp_path / "env"
    env_dir.mkdir()
    monkeypatch.setattr(presets, "PRESETS_DIR", overrides)
    monkeypatch.setenv("TREND_PRESETS_DIR", str(env_dir))

    dirs = presets._candidate_preset_dirs()
    assert dirs[:2] == (overrides, env_dir)


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_preset_registry_loads_presets_and_warns_on_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    base = tmp_path / "base"
    base.mkdir()
    env_dir = tmp_path / "env"
    env_dir.mkdir()

    _write_yaml(
        base / "trend.yml",
        """
name: Base Trend
signals:
  window: 63
  lag: 1
  vol_adjust: true
metrics:
  sharpe: 1.0
        """.strip(),
    )

    _write_yaml(
        env_dir / "trend.yml",
        """
name: Env Trend
signals:
  window: 90
  lag: 3
metrics:
  sharpe: 0.5
        """.strip(),
    )

    _write_yaml(base / "empty.yml", "[]")

    monkeypatch.setattr(presets, "PRESETS_DIR", base)
    monkeypatch.setenv("TREND_PRESETS_DIR", str(env_dir))

    with caplog.at_level(logging.WARNING):
        registry = presets._preset_registry()
    assert "Duplicate trend preset slug" in caplog.text
    assert registry["trend"].label == "Env Trend"
    assert registry["trend"].trend_spec.window == 90
    assert "empty" not in registry

    assert presets.list_preset_slugs() == ("trend",)
    presets_list = presets.list_trend_presets()
    assert [p.label for p in presets_list] == ["Env Trend"]
    assert presets.get_trend_preset("trend").label == "Env Trend"
    assert presets.get_trend_preset("Env Trend").slug == "trend"


def test_get_trend_preset_invalid_name_raises():
    with pytest.raises(KeyError):
        presets.get_trend_preset("")
    with pytest.raises(KeyError):
        presets.get_trend_preset("unknown")


def test_metric_key_helpers_handle_missing():
    assert presets.normalise_metric_key("Sharpe_Ratio") == "sharpe"
    assert presets.normalise_metric_key("") is None
    assert presets.pipeline_metric_key("volatility") == "Volatility"
    assert presets.pipeline_metric_key(None) is None  # type: ignore[arg-type]


def test_apply_trend_preset_merges_into_config(sample_preset: presets.TrendPreset):
    class DummyConfig:
        def __init__(self) -> None:
            self.signals = {"existing": "value"}
            self.vol_adjust = {"enabled": True, "extra": "keep"}
            self.run = {"other": 1}

    config = DummyConfig()
    presets.apply_trend_preset(config, sample_preset)

    assert config.signals["kind"] == sample_preset.trend_spec.kind
    assert config.signals["existing"] == "value"
    assert config.vol_adjust["enabled"] is False
    assert config.vol_adjust["target_vol"] == pytest.approx(0.7)
    assert config.vol_adjust["extra"] == "keep"
    assert config.run["trend_preset"] == "momentum"


def test_apply_trend_preset_handles_non_mapping_sections(
    sample_preset: presets.TrendPreset,
):
    class DummyConfig:
        def __init__(self) -> None:
            self.signals = None
            self.vol_adjust = "disabled"
            self.run = None

    config = DummyConfig()
    presets.apply_trend_preset(config, sample_preset)

    assert isinstance(config.signals, dict)
    assert config.signals["window"] == sample_preset.trend_spec.window
    assert config.vol_adjust["enabled"] is False
    assert config.run["trend_preset"] == sample_preset.slug


def test_apply_trend_preset_handles_mappingproxy_sections(
    sample_preset: presets.TrendPreset,
):
    from types import MappingProxyType

    class DummyConfig:
        def __init__(self) -> None:
            self.signals = MappingProxyType({"kind": "other"})
            self.vol_adjust = MappingProxyType({"enabled": True})
            self.run = MappingProxyType({})

    config = DummyConfig()
    presets.apply_trend_preset(config, sample_preset)

    assert config.signals["kind"] == "tsmom"
    assert config.vol_adjust["enabled"] is sample_preset.trend_spec.vol_adjust
    assert config.run["trend_preset"] == sample_preset.slug
