import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from trend_analysis import presets
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


def test_helper_functions_cover_edge_cases(sample_config: dict[str, Any]):
    frozen = presets._freeze_mapping(sample_config)
    assert dict(frozen) == sample_config
    with pytest.raises(TypeError):
        frozen["new"] = 1  # type: ignore[index]

    weights = presets._normalise_metric_weights(sample_config["metrics"])
    assert weights == {"sharpe": 2.0}
    assert presets._normalise_metric_weights({"sharpe": object()}) == {}

    assert presets._coerce_int("10", default=3, minimum=1) == 10
    assert presets._coerce_int("bad", default=5, minimum=7) == 5
    assert presets._coerce_optional_int("12", minimum=5) == 12
    assert presets._coerce_optional_int("2", minimum=5) is None
    assert presets._coerce_optional_int(None) is None
    assert presets._coerce_optional_int("bad") is None
    assert presets._coerce_optional_float("0.7", minimum=0.5) == pytest.approx(0.7)
    assert presets._coerce_optional_float("0.2", minimum=0.5) is None
    assert presets._coerce_optional_float("bad") is None


def test_build_trend_spec_and_preset_defaults(sample_config: dict[str, Any]):
    spec = presets._build_trend_spec(sample_config)
    assert isinstance(spec, TrendSpec)
    assert spec.window == 126
    assert spec.min_periods == 126  # clamped to window
    assert spec.vol_target == pytest.approx(0.7)

    preset = presets.TrendPreset(
        slug="momentum",
        label="Momentum",
        description="Long-term trend following",
        trend_spec=spec,
        _config=presets._freeze_mapping(sample_config),
    )

    defaults = preset.form_defaults()
    assert defaults == {
        "lookback_months": 48,
        "rebalance_frequency": "Weekly",
        "min_track_months": 12,
        "selection_count": 25,
        "risk_target": pytest.approx(0.33),
        "weighting_scheme": "risk",
        "cooldown_months": 6,
        "metrics": {"sharpe": 2.0},
    }

    mapping = preset.signals_mapping()
    assert mapping == {
        "kind": "tsmom",
        "window": 126,
        "lag": 2,
        "vol_adjust": False,
        "zscore": False,
        "min_periods": 126,
        "vol_target": pytest.approx(0.7),
    }

    pipeline_weights = preset.metrics_pipeline()
    assert pipeline_weights == {"Sharpe": 2.0}


def test_build_trend_spec_handles_non_mapping_signals() -> None:
    spec = presets._build_trend_spec({"signals": [1, 2, 3]})
    assert spec.window == 63
    assert spec.min_periods is None
    assert spec.vol_target is None


def test_form_defaults_handles_non_mapping_portfolio(
    sample_config: dict[str, Any],
) -> None:
    config = dict(sample_config)
    config["portfolio"] = "not-a-mapping"
    preset = presets.TrendPreset(
        slug="fallback",
        label="Fallback",
        description="",
        trend_spec=presets._build_trend_spec(config),
        _config=presets._freeze_mapping(config),
    )

    defaults = preset.form_defaults()
    assert defaults["weighting_scheme"] == "equal"
    assert defaults["cooldown_months"] == 3


def test_signals_mapping_omits_optional_fields() -> None:
    preset = presets.TrendPreset(
        slug="base",
        label="Base",
        description="",
        trend_spec=presets._build_trend_spec({"signals": {}}),
        _config=presets._freeze_mapping({"signals": {}}),
    )

    mapping = preset.signals_mapping()
    assert "min_periods" not in mapping
    assert "vol_target" not in mapping


def test_vol_adjust_defaults_cover_branches(sample_config: dict[str, Any]):
    base_spec = presets._build_trend_spec(sample_config)
    preset = presets.TrendPreset(
        slug="vol",
        label="Vol",
        description="",
        trend_spec=base_spec,
        _config=presets._freeze_mapping(sample_config),
    )

    defaults = preset.vol_adjust_defaults()
    assert defaults["enabled"] is False
    assert defaults["target_vol"] == pytest.approx(0.7)
    assert defaults["window"]["length"] == 21

    overrides = dict(sample_config)
    overrides["vol_adjust"] = MappingProxyType({"enabled": True, "window": {}})
    overrides["signals"] = MappingProxyType({"vol_adjust": True, "window": 42})
    custom = presets.TrendPreset(
        slug="override",
        label="Override",
        description="",
        trend_spec=presets._build_trend_spec(overrides),
        _config=presets._freeze_mapping(overrides),
    )
    custom_defaults = custom.vol_adjust_defaults()
    assert custom_defaults["enabled"] is True
    assert custom_defaults["window"]["length"] == 42

    signals_only = presets.TrendPreset(
        slug="signals",
        label="Signals",
        description="",
        trend_spec=presets._build_trend_spec({"signals": {"vol_adjust": True}}),
        _config=presets._freeze_mapping({"signals": {"vol_adjust": True}}),
    )
    signals_defaults = signals_only.vol_adjust_defaults()
    assert signals_defaults["enabled"] is True

    spec_override = presets.TrendPreset(
        slug="spec",
        label="Spec",
        description="",
        trend_spec=TrendSpec(vol_adjust=True, window=21, lag=1, min_periods=None),
        _config=presets._freeze_mapping({}),
    )
    spec_defaults = spec_override.vol_adjust_defaults()
    assert spec_defaults["enabled"] is True
    assert spec_defaults["window"]["length"] == 21

    empty_config = presets.TrendPreset(
        slug="empty",
        label="Empty",
        description="",
        trend_spec=TrendSpec(),
        _config=presets._freeze_mapping({}),
    )
    empty_defaults = empty_config.vol_adjust_defaults()
    assert empty_defaults["enabled"] is True
    assert (
        empty_defaults["window"]["length"]
        == presets._DEFAULT_VOL_ADJUST["window"]["length"]
    )

    none_target = presets.TrendPreset(
        slug="none-target",
        label="NoneTarget",
        description="",
        trend_spec=TrendSpec(),
        _config=presets._freeze_mapping({"vol_adjust": {"target_vol": None}}),
    )
    none_defaults = none_target.vol_adjust_defaults()
    assert none_defaults["target_vol"] == presets._DEFAULT_VOL_ADJUST["target_vol"]

    proxy_window = presets.TrendPreset(
        slug="proxy-window",
        label="ProxyWindow",
        description="",
        trend_spec=TrendSpec(window=63),
        _config=presets._freeze_mapping(
            {
                "vol_adjust": MappingProxyType(
                    {"window": MappingProxyType({"length": None})}
                )
            }
        ),
    )
    proxy_defaults = proxy_window.vol_adjust_defaults()
    assert (
        proxy_defaults["window"]["length"]
        == presets._DEFAULT_VOL_ADJUST["window"]["length"]
    )


def test_apply_trend_preset_merges_sections(sample_config: dict[str, Any]):
    preset = presets.TrendPreset(
        slug="merge",
        label="Merge",
        description="",
        trend_spec=presets._build_trend_spec(sample_config),
        _config=presets._freeze_mapping(sample_config),
    )

    class DummyConfig:
        def __init__(self) -> None:
            self.signals = {"existing": "value"}
            self.vol_adjust = MappingProxyType({"extra": "keep"})
            self.run = None

    config = DummyConfig()
    presets.apply_trend_preset(config, preset)
    assert config.signals["kind"] == preset.trend_spec.kind
    assert config.signals["existing"] == "value"
    assert config.vol_adjust["target_vol"] == pytest.approx(0.7)
    assert config.vol_adjust["extra"] == "keep"
    assert config.run["trend_preset"] == "merge"

    class EmptyConfig:
        def __init__(self) -> None:
            self.signals = None
            self.vol_adjust = "disabled"
            self.run = {}

    empty = EmptyConfig()
    presets.apply_trend_preset(empty, preset)
    assert empty.signals["window"] == preset.trend_spec.window
    assert empty.vol_adjust["enabled"] is False

    class MutableConfig:
        def __init__(self) -> None:
            self.signals = {"kind": "other"}
            self.vol_adjust = {"enabled": True}
            self.run = MappingProxyType({"existing": "value"})

    mutable = MutableConfig()
    presets.apply_trend_preset(mutable, preset)
    assert mutable.vol_adjust["enabled"] is False
    assert mutable.run["trend_preset"] == preset.slug


def test_registry_and_lookup_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    base = tmp_path / "base"
    base.mkdir()
    env_dir = tmp_path / "env"
    env_dir.mkdir()

    (base / "alpha.yml").write_text(
        """name: Alpha
signals:
  window: 50
  lag: 2
metrics:
  sharpe: 1.0
""",
        encoding="utf-8",
    )

    (env_dir / "alpha.yml").write_text(
        """name: Alpha Override
signals:
  window: 80
  lag: 3
""",
        encoding="utf-8",
    )

    (base / "empty.yml").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(presets, "PRESETS_DIR", base)
    monkeypatch.setenv("TREND_PRESETS_DIR", str(env_dir))

    dirs = presets._candidate_preset_dirs()
    assert dirs[:2] == (base, env_dir)

    monkeypatch.setenv("TREND_PRESETS_DIR", str(base))
    duplicate_dirs = presets._candidate_preset_dirs()
    assert duplicate_dirs[0] == base

    monkeypatch.setattr(presets, "PRESETS_DIR", presets._DEFAULT_PRESETS_DIR)
    monkeypatch.setenv("TREND_PRESETS_DIR", str(base))
    defaults_dirs = presets._candidate_preset_dirs()
    assert presets._DEFAULT_PRESETS_DIR in defaults_dirs
    assert base in defaults_dirs

    monkeypatch.setattr(presets, "PRESETS_DIR", base)
    monkeypatch.setenv("TREND_PRESETS_DIR", str(env_dir))

    with caplog.at_level(logging.WARNING):
        registry = presets._preset_registry()
    assert "Duplicate trend preset slug" in caplog.text

    preset = registry["alpha"]
    assert preset.label == "Alpha Override"
    assert preset.trend_spec.window == 80

    slugs = presets.list_preset_slugs()
    assert slugs == ("alpha",)

    presets_list = presets.list_trend_presets()
    assert [p.slug for p in presets_list] == ["alpha"]
    assert presets.get_trend_preset("alpha") is preset
    assert presets.get_trend_preset("Alpha Override") is preset
    assert presets.normalise_metric_key("Sharpe_Ratio") == "sharpe"
    assert presets.normalise_metric_key("") is None
    assert presets.pipeline_metric_key("volatility") == "Volatility"
    assert presets.pipeline_metric_key(None) is None  # type: ignore[arg-type]

    with pytest.raises(KeyError):
        presets.get_trend_preset("")
    with pytest.raises(KeyError):
        presets.get_trend_preset("missing")


def test_load_yaml_handles_non_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    base = tmp_path / "yaml"
    base.mkdir()
    path = base / "invalid.yml"
    path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(presets, "PRESETS_DIR", base)
    assert presets._load_yaml(path) == {}
