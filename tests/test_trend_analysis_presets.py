from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import yaml

from trend_analysis.presets import (
    PIPELINE_METRIC_ALIASES,
    UI_METRIC_ALIASES,
    TrendPreset,
    _build_trend_spec,
    _coerce_int,
    _coerce_optional_float,
    _coerce_optional_int,
    _load_yaml,
    _freeze_mapping,
    _normalise_metric_weights,
    apply_trend_preset,
    get_trend_preset,
    list_preset_slugs,
    list_trend_presets,
    normalise_metric_key,
    pipeline_metric_key,
)


@pytest.fixture(autouse=True)
def clear_preset_cache():
    import importlib

    module = importlib.import_module("trend_analysis.presets")
    module._preset_registry.cache_clear()
    yield
    module._preset_registry.cache_clear()


def test_freeze_mapping_creates_read_only_copy():
    data = {"a": 1, "b": 2}
    frozen = _freeze_mapping(data)
    assert isinstance(frozen, MappingProxyType)
    data["c"] = 3
    assert "c" not in frozen


def test_normalise_metric_weights_filters_invalid_entries():
    weights = _normalise_metric_weights({"Sharpe": "2", "invalid": "x", "vol": 1})
    assert weights == {"sharpe": 2.0, "vol": 1.0}


def test_normalise_metric_weights_ignores_non_numeric_values():
    weights = _normalise_metric_weights({"sharpe": "not-a-number"})
    assert weights == {}


@pytest.mark.parametrize(
    "value, default, minimum, expected",
    [("10", 5, 1, 10), ("-1", 5, 1, 1), (None, 5, 0, 5)],
)
def test_coerce_int(value, default, minimum, expected):
    assert _coerce_int(value, default, minimum) == expected


@pytest.mark.parametrize(
    "value, minimum, expected",
    [(None, 1, None), ("5", 1, 5), ("-1", 1, None), ("abc", 1, None)],
)
def test_coerce_optional_int(value, minimum, expected):
    assert _coerce_optional_int(value, minimum) == expected


@pytest.mark.parametrize(
    "value, minimum, expected",
    [(None, 0.0, None), ("0.5", 0.0, 0.5), ("-0.1", 0.0, None), ("abc", 0.0, None)],
)
def test_coerce_optional_float(value, minimum, expected):
    assert _coerce_optional_float(value, minimum) == expected


def test_build_trend_spec_normalises_min_periods():
    spec = _build_trend_spec({
        "signals": {
            "window": 20,
            "min_periods": 30,
            "lag": 2,
            "vol_adjust": True,
            "vol_target": "0.1",
            "zscore": True,
        }
    })
    assert spec.window == 20
    assert spec.min_periods == 20
    assert spec.vol_adjust is True
    assert spec.vol_target == 0.1


def test_build_trend_spec_with_missing_mapping():
    spec = _build_trend_spec({"signals": None})
    assert spec.window == 63
    assert spec.min_periods is None


def test_trend_preset_helpers():
    spec = _build_trend_spec(
        {"signals": {"window": 15, "min_periods": 10, "vol_target": "0.2"}}
    )
    config = {
        "lookback_months": "12",
        "rebalance_frequency": "weekly",
        "min_track_months": "3",
        "selection_count": "5",
        "risk_target": "0.25",
        "metrics": {"sharpe_ratio": "2", "volatility": "0.5"},
        "portfolio": "not-a-mapping",
        "vol_adjust": {
            "enabled": False,
            "window": MappingProxyType({"length": 10}),
        },
    }
    preset = TrendPreset(
        slug="demo",
        label="Demo",
        description="desc",
        trend_spec=spec,
        _config=_freeze_mapping(config),
    )

    defaults = preset.form_defaults()
    assert defaults["lookback_months"] == 12
    assert defaults["metrics"] == {"sharpe": 2.0, "vol": 0.5}

    mapping = preset.signals_mapping()
    assert mapping["window"] == 15
    assert mapping["lag"] == 1
    assert mapping["min_periods"] == 10
    assert mapping["vol_target"] == 0.2

    vol_defaults = preset.vol_adjust_defaults()
    assert vol_defaults["enabled"] is False
    assert vol_defaults["window"]["length"] == 10

    pipeline_metrics = preset.metrics_pipeline()
    assert pipeline_metrics == {"Sharpe": 2.0, "Volatility": 0.5}


def test_preset_registry_reads_files(tmp_path: Path, monkeypatch):
    config_dir = tmp_path / "presets"
    config_dir.mkdir()
    payload = {
        "name": "Alpha",
        "description": "First",
        "signals": {"window": 30},
        "metrics": {"sharpe": 1},
    }
    (config_dir / "alpha.yml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    (config_dir / "beta.yml").write_text(yaml.safe_dump({"signals": {"window": 10}}), encoding="utf-8")

    monkeypatch.setattr("trend_analysis.presets.PRESETS_DIR", config_dir)

    presets = list_trend_presets()
    assert [p.slug for p in presets] == ["alpha", "beta"]

    slugs = list_preset_slugs()
    assert slugs == ("alpha", "beta")

    preset = get_trend_preset("alpha")
    assert preset.label == "Alpha"
    assert get_trend_preset("Alpha").slug == "alpha"

    with pytest.raises(KeyError):
        get_trend_preset("missing")


def test_preset_registry_handles_missing_directory(monkeypatch):
    empty_dir = Path("/tmp/nonexistent-presets-dir")
    monkeypatch.setattr("trend_analysis.presets.PRESETS_DIR", empty_dir)
    assert list_trend_presets() == ()
    assert list_preset_slugs() == ()


def test_preset_registry_skips_empty_files(tmp_path: Path, monkeypatch):
    config_dir = tmp_path / "presets"
    config_dir.mkdir()
    (config_dir / "empty.yml").write_text("null", encoding="utf-8")
    monkeypatch.setattr("trend_analysis.presets.PRESETS_DIR", config_dir)
    assert list_preset_slugs() == ()


def test_apply_trend_preset_updates_config(tmp_path: Path, monkeypatch):
    config_dir = tmp_path / "presets"
    config_dir.mkdir()
    data = {
        "name": "Alpha",
        "signals": {"window": 12, "vol_adjust": True},
        "vol_adjust": {"target_vol": 0.3},
    }
    (config_dir / "alpha.yml").write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr("trend_analysis.presets.PRESETS_DIR", config_dir)

    preset = get_trend_preset("alpha")
    target = SimpleNamespace(signals={"lag": 4}, vol_adjust={}, run={})
    apply_trend_preset(target, preset)

    assert target.signals["window"] == 12
    assert target.signals["lag"] == 1
    assert target.vol_adjust["target_vol"] == 0.3
    assert target.run["trend_preset"] == "alpha"


def test_apply_trend_preset_with_non_mapping_sections(tmp_path: Path, monkeypatch):
    config_dir = tmp_path / "presets"
    config_dir.mkdir()
    data = {"signals": {"window": 8}}
    (config_dir / "alpha.yml").write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr("trend_analysis.presets.PRESETS_DIR", config_dir)

    preset = get_trend_preset("alpha")
    target = SimpleNamespace(signals=None, vol_adjust=MappingProxyType({"enabled": True}), run=())
    apply_trend_preset(target, preset)

    assert isinstance(target.signals, dict)
    assert target.vol_adjust["enabled"] is False
    assert target.run["trend_preset"] == "alpha"


def test_apply_trend_preset_with_none_sections(tmp_path: Path, monkeypatch):
    config_dir = tmp_path / "presets"
    config_dir.mkdir()
    data = {"signals": {"window": 6}}
    (config_dir / "alpha.yml").write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr("trend_analysis.presets.PRESETS_DIR", config_dir)

    preset = get_trend_preset("alpha")
    target = SimpleNamespace(signals=None, vol_adjust=None, run=MappingProxyType({"legacy": True}))
    apply_trend_preset(target, preset)

    assert target.signals["window"] == 6
    assert target.vol_adjust["enabled"] is False
    assert "legacy" not in target.vol_adjust
    assert target.run["trend_preset"] == "alpha"


@pytest.mark.parametrize(
    "name, expected",
    [("Sharpe", "sharpe"), ("", None), ("unknown", None)],
)
def test_normalise_metric_key(name, expected):
    assert normalise_metric_key(name) == expected


@pytest.mark.parametrize(
    "name, expected",
    [("sharpe", "Sharpe"), ("volatility", "Volatility"), ("", None)],
)
def test_pipeline_metric_key(name, expected):
    assert pipeline_metric_key(name) == expected


def test_alias_tables_are_mapping_proxy():
    assert isinstance(UI_METRIC_ALIASES, MappingProxyType)
    assert isinstance(PIPELINE_METRIC_ALIASES, MappingProxyType)


def test_load_yaml_returns_empty_for_invalid(tmp_path: Path):
    yaml_path = tmp_path / "invalid.yml"
    yaml_path.write_text("- item", encoding="utf-8")
    data = _load_yaml(yaml_path)
    assert data == {}


def test_get_trend_preset_requires_name():
    with pytest.raises(KeyError):
        get_trend_preset("")
