"""Focused coverage for :mod:`trend_analysis.presets`."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import yaml

from trend_analysis import presets
from trend_analysis.presets import (
    PIPELINE_METRIC_ALIASES,
    UI_METRIC_ALIASES,
    TrendPreset,
    _build_trend_spec,
    _coerce_int,
    _coerce_optional_float,
    _coerce_optional_int,
    _freeze_mapping,
    _normalise_metric_weights,
    apply_trend_preset,
    get_trend_preset,
    list_preset_slugs,
    list_trend_presets,
    normalise_metric_key,
    pipeline_metric_key,
)
from trend_analysis.signals import TrendSpec


def write_preset(path: Path, **fields: Any) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(fields, handle, sort_keys=False)
    return path


def create_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preset_dir = tmp_path / "presets"
    preset_dir.mkdir()
    write_preset(
        preset_dir / "alpha.yml",
        name="Alpha",
        description="Alpha preset",
        lookback_months=12,
        min_track_months=6,
        selection_count=5,
        risk_target=0.2,
        metrics={"sharpe_ratio": 0.5, "INVALID": "skip"},
        signals={
            "window": 80,
            "min_periods": 120,
            "lag": 2,
            "vol_adjust": True,
            "vol_target": 0.15,
            "zscore": True,
        },
        portfolio={"weighting_scheme": "risk_parity", "cooldown_months": 2},
        vol_adjust={"window": {"length": 20}},
    )
    write_preset(
        preset_dir / "zulu.yml",
        name="Zulu",
        description="Zulu preset",
        lookback_months=36,
        metrics={"return_ann": 0.7},
        signals={"window": 63, "lag": 1, "vol_adjust": False, "zscore": False},
    )
    write_preset(preset_dir / "empty.yml")

    monkeypatch.setattr(presets, "PRESETS_DIR", preset_dir)
    presets._preset_registry.cache_clear()


def test_preset_registry_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_registry(tmp_path, monkeypatch)

    registry = presets._preset_registry()
    assert set(registry) == {"alpha", "zulu"}
    alpha = registry["alpha"]
    assert isinstance(alpha, TrendPreset)
    assert alpha.trend_spec.window == 80
    assert alpha.trend_spec.min_periods == 80  # clamped to window
    assert alpha.trend_spec.vol_adjust is True
    assert alpha.trend_spec.vol_target == 0.15
    assert alpha.form_defaults()["risk_target"] == 0.2

    # Ensure empty preset is skipped
    assert "empty" not in registry


def test_list_helpers_sorted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    create_registry(tmp_path, monkeypatch)

    slugs = list_preset_slugs()
    assert slugs == ("alpha", "zulu")

    labels = tuple(p.label for p in list_trend_presets())
    assert labels == ("Alpha", "Zulu")


def test_get_trend_preset_by_slug_and_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_registry(tmp_path, monkeypatch)

    alpha_by_slug = get_trend_preset("alpha")
    alpha_by_label = get_trend_preset("ALPHA")
    assert alpha_by_slug is alpha_by_label

    with pytest.raises(KeyError):
        get_trend_preset("")
    with pytest.raises(KeyError):
        get_trend_preset("unknown")


def test_trend_preset_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    create_registry(tmp_path, monkeypatch)
    preset = get_trend_preset("alpha")

    defaults = preset.form_defaults()
    assert defaults["lookback_months"] == 12
    assert defaults["min_track_months"] == 6
    assert defaults["selection_count"] == 5
    assert defaults["metrics"] == {"sharpe": 0.5}

    signals_mapping = preset.signals_mapping()
    assert signals_mapping == {
        "kind": preset.trend_spec.kind,
        "window": 80,
        "lag": 2,
        "vol_adjust": True,
        "zscore": True,
        "min_periods": 80,
        "vol_target": 0.15,
    }

    vol_defaults = preset.vol_adjust_defaults()
    assert vol_defaults["enabled"] is True
    assert vol_defaults["window"]["length"] == 20
    assert vol_defaults["window"]["length"] != preset.trend_spec.window

    metrics_pipeline = preset.metrics_pipeline()
    assert metrics_pipeline["Sharpe"] == defaults["metrics"]["sharpe"]


def test_vol_adjust_defaults_handles_mapping_proxy() -> None:
    spec = TrendSpec(
        window=42,
        min_periods=None,
        lag=2,
        vol_adjust=False,
        vol_target=None,
        zscore=False,
    )
    proxy_window = MappingProxyType({"length": 7})
    config = _freeze_mapping({"vol_adjust": MappingProxyType({"window": proxy_window})})
    preset = TrendPreset(
        slug="custom",
        label="Custom",
        description="",
        trend_spec=spec,
        _config=config,
    )

    defaults = preset.vol_adjust_defaults()
    assert defaults["enabled"] is False
    assert defaults["target_vol"] is None
    assert defaults["window"] == {"length": 7}


def test_vol_adjust_defaults_sets_missing_values() -> None:
    spec = TrendSpec(
        window=55,
        min_periods=10,
        lag=1,
        vol_adjust=True,
        vol_target=0.3,
        zscore=False,
    )
    preset = TrendPreset(
        slug="growth",
        label="Growth",
        description="",
        trend_spec=spec,
        _config=_freeze_mapping({}),
    )

    defaults = preset.vol_adjust_defaults()
    assert defaults["enabled"] is True
    assert defaults["target_vol"] == 0.3
    assert defaults["window"] == {"length": 55}


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
    apply_trend_preset(config, preset)

    assert config.signals["window"] == 63
    assert config.vol_adjust["enabled"] is False
    assert config.run["trend_preset"] == "zulu"

    empty = SimpleNamespace()
    apply_trend_preset(empty, preset)
    assert empty.signals["window"] == 63
    assert empty.vol_adjust["enabled"] is False
    assert empty.run["trend_preset"] == "zulu"


def test_apply_trend_preset_branch_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_registry(tmp_path, monkeypatch)
    preset = get_trend_preset("alpha")

    config = SimpleNamespace(signals=None, vol_adjust=42, run="not-a-mapping")
    apply_trend_preset(config, preset)

    assert config.signals["window"] == preset.trend_spec.window
    assert config.vol_adjust["window"]["length"] == 20
    assert config.run["trend_preset"] == "alpha"

    mapping_config = SimpleNamespace(
        signals=None,
        vol_adjust=MappingProxyType({"window": {"length": 10}}),
        run=MappingProxyType({"existing": "value"}),
    )
    apply_trend_preset(mapping_config, preset)
    assert mapping_config.vol_adjust["window"]["length"] == 20
    assert mapping_config.run["trend_preset"] == "alpha"


def test_metric_alias_helpers() -> None:
    assert normalise_metric_key("Sharpe_Ratio") == "sharpe"
    assert normalise_metric_key("") is None
    assert pipeline_metric_key("max_drawdown") == "MaxDrawdown"
    assert pipeline_metric_key("unknown") is None
    assert pipeline_metric_key("") is None


def test_candidate_dirs_include_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    presets._preset_registry.cache_clear()
    monkeypatch.delenv("TREND_PRESETS_DIR", raising=False)
    monkeypatch.setattr(presets, "PRESETS_DIR", presets._DEFAULT_PRESETS_DIR)

    candidates = presets._candidate_preset_dirs()
    assert presets._DEFAULT_PRESETS_DIR in candidates
    assert len(candidates) >= 1


def test_candidate_dirs_env_override_and_deduplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir = tmp_path / "presets"
    base_dir.mkdir()
    monkeypatch.setattr(presets, "PRESETS_DIR", base_dir)
    monkeypatch.setenv("TREND_PRESETS_DIR", str(base_dir))
    presets._preset_registry.cache_clear()

    candidates = presets._candidate_preset_dirs()
    assert candidates == (base_dir,)


def test_get_trend_preset_matches_display_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preset_dir = tmp_path / "presets"
    preset_dir.mkdir()
    write_preset(
        preset_dir / "growth_fund.yml",
        name="Growth Fund",
        signals={"window": 20, "lag": 1, "vol_adjust": False, "zscore": False},
    )
    monkeypatch.setattr(presets, "PRESETS_DIR", preset_dir)
    presets._preset_registry.cache_clear()

    preset = get_trend_preset("Growth Fund")
    assert preset.slug == "growth_fund"

    # Ensure alias tables remain immutable
    with pytest.raises(TypeError):
        UI_METRIC_ALIASES["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        PIPELINE_METRIC_ALIASES["new"] = "value"  # type: ignore[index]


def test_preset_registry_warns_on_duplicate_slug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    base_dir = tmp_path / "base"
    override_dir = tmp_path / "override"
    base_dir.mkdir()
    override_dir.mkdir()

    write_preset(
        base_dir / "alpha.yml",
        name="Alpha",
        signals={"window": 20, "lag": 1, "vol_adjust": False, "zscore": False},
    )
    write_preset(
        override_dir / "alpha.yml",
        name="Alpha Override",
        signals={"window": 25, "lag": 1, "vol_adjust": False, "zscore": False},
    )

    monkeypatch.setattr(presets, "PRESETS_DIR", base_dir)
    monkeypatch.setenv("TREND_PRESETS_DIR", str(override_dir))
    presets._preset_registry.cache_clear()

    with caplog.at_level("WARNING"):
        registry = presets._preset_registry()

    assert "Duplicate trend preset slug 'alpha'" in caplog.text
    # Override should win due to search order
    assert registry["alpha"].label == "Alpha Override"

    presets._preset_registry.cache_clear()


def test_registry_empty_when_directory_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(presets, "PRESETS_DIR", Path("/nonexistent"))
    presets._preset_registry.cache_clear()
    assert not presets._preset_registry()


def test_load_yaml_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    assert presets._load_yaml(path) == {}


def test_build_trend_spec_defaults() -> None:
    spec = _build_trend_spec({"signals": ["not", "mapping"]})
    assert spec.window == 63
    assert spec.min_periods is None


def test_internal_helper_functions() -> None:
    assert _coerce_int("5", default=1, minimum=3) == 5
    assert _coerce_int(0, default=5, minimum=3) == 3
    assert _coerce_int("bad", default=7, minimum=1) == 7

    assert _coerce_optional_int(None) is None
    assert _coerce_optional_int("4") == 4
    assert _coerce_optional_int("bad") is None
    assert _coerce_optional_int(0, minimum=1) is None

    assert _coerce_optional_float(None) is None
    assert _coerce_optional_float("0.2") == 0.2
    assert _coerce_optional_float("bad") is None
    assert _coerce_optional_float(-0.1, minimum=0.0) is None

    weights = _normalise_metric_weights(
        {"Sharpe": "2", "invalid": object(), "return_ann": "bad"}
    )
    assert weights == {"sharpe": 2.0}

    frozen = _freeze_mapping({"key": 1})
    with pytest.raises(TypeError):
        frozen["key"] = 2  # type: ignore[index]


def test_vol_adjust_defaults_and_metrics_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_registry(tmp_path, monkeypatch)
    zulu = get_trend_preset("zulu")

    defaults = zulu.vol_adjust_defaults()
    assert defaults["enabled"] is False
    assert defaults["window"]["length"] == zulu.trend_spec.window
    assert defaults["target_vol"] is None

    pipeline = zulu.metrics_pipeline()
    assert pipeline["AnnualReturn"] == pytest.approx(0.7)
