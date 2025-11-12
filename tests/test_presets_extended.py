import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from trend_analysis import presets
from trend_analysis.signals import TrendSpec


def _reset_registry() -> None:
    presets._preset_registry.cache_clear()


def test_normalise_helpers_and_build_trend_spec() -> None:
    raw = {"Sharpe": "1.5", "invalid": "nan", "": 10, "volatility": "foo"}
    weights = presets._normalise_metric_weights(raw)
    assert weights == {"sharpe": 1.5}

    assert presets._coerce_int("7", default=3, minimum=5) == 7
    # Fallback when coercion fails should use provided default
    assert presets._coerce_int("foo", default=3, minimum=5) == 3

    assert presets._coerce_optional_int("11", minimum=8) == 11
    assert presets._coerce_optional_int("bad", minimum=8) is None
    assert presets._coerce_optional_int("3", minimum=5) is None

    assert presets._coerce_optional_float("0.75", minimum=0.5) == pytest.approx(0.75)
    assert presets._coerce_optional_float("-1", minimum=0.5) is None
    # Type errors should be swallowed and return None
    assert presets._coerce_optional_float(object()) is None
    assert presets._coerce_optional_float(None) is None

    spec = presets._build_trend_spec(
        {
            "signals": {
                "window": "12",
                "min_periods": 40,
                "lag": "2",
                "vol_adjust": "yes",
                "vol_target": "0.25",
                "zscore": 1,
            }
        }
    )
    assert isinstance(spec, TrendSpec)
    # min_periods cannot exceed window
    assert spec.window == 12
    assert spec.min_periods == 12
    assert spec.lag == 2
    assert spec.vol_adjust is True
    assert spec.vol_target == pytest.approx(0.25)
    assert spec.zscore is True


def test_trend_preset_helpers_cover_defaults() -> None:
    spec = TrendSpec(
        window=20, min_periods=10, vol_adjust=True, vol_target=0.2, zscore=True
    )
    preset_cfg = {
        "lookback_months": "18",
        "rebalance_frequency": "quarterly",
        "min_track_months": "bad",
        "selection_count": "",
        "risk_target": "0.35",
        "portfolio": "not-a-mapping",
        "metrics": {"sharpe": "2", "volatility": 0.5},
        "signals": {"vol_adjust": False},
        "vol_adjust": {"enabled": None, "target_vol": None},
    }
    preset = presets.TrendPreset(
        slug="demo",
        label="Demo",
        description="desc",
        trend_spec=spec,
        _config=presets._freeze_mapping(preset_cfg),
    )

    defaults = preset.form_defaults()
    assert defaults["lookback_months"] == 18
    # min_track_months falls back to minimum because coercion fails
    assert defaults["min_track_months"] == 24
    assert defaults["selection_count"] == 10
    assert defaults["risk_target"] == pytest.approx(0.35)
    assert defaults["metrics"] == {"sharpe": 2.0, "vol": 0.5}
    # Non-mapping portfolio inputs should be normalised to defaults
    assert defaults["weighting_scheme"] == "equal"

    signals = preset.signals_mapping()
    assert signals["min_periods"] == 10
    assert signals["vol_target"] == pytest.approx(0.2)


def test_trend_preset_portfolio_mapping_branch() -> None:
    spec = TrendSpec()
    preset_cfg = {
        "portfolio": {"weighting_scheme": "custom", "cooldown_months": 4},
    }
    preset = presets.TrendPreset(
        slug="portfolio",
        label="Portfolio",
        description="",
        trend_spec=spec,
        _config=presets._freeze_mapping(preset_cfg),
    )
    defaults = preset.form_defaults()
    assert defaults["weighting_scheme"] == "custom"
    assert defaults["cooldown_months"] == 4


def test_apply_trend_preset_updates_namespace() -> None:
    spec = TrendSpec(window=15, vol_adjust=False)
    preset = presets.TrendPreset(
        slug="guard",
        label="Guard",
        description="",
        trend_spec=spec,
        _config=presets._freeze_mapping({}),
    )
    cfg = SimpleNamespace(signals={"window": 5}, vol_adjust=("ignored",), run={})
    presets.apply_trend_preset(cfg, preset)
    assert cfg.signals["window"] == 15
    assert cfg.vol_adjust["enabled"] is True
    assert cfg.run["trend_preset"] == "guard"


def test_vol_adjust_defaults_handles_signal_overrides() -> None:
    spec = TrendSpec(window=40, vol_adjust=False)
    preset_cfg = {
        "signals": {"vol_adjust": True},
    }
    preset = presets.TrendPreset(
        slug="vol",
        label="Vol",
        description="",
        trend_spec=spec,
        _config=presets._freeze_mapping(preset_cfg),
    )
    defaults = preset.vol_adjust_defaults()
    # enabled should inherit from signals mapping when not explicitly set
    assert defaults["enabled"] is True
    # target_vol should fall back to module defaults when spec lacks override
    assert defaults["target_vol"] == presets._DEFAULT_VOL_ADJUST["target_vol"]
    assert defaults["window"]["length"] == 40


def test_vol_adjust_defaults_copies_window_mapping() -> None:
    spec = TrendSpec(window=20, vol_adjust=True, vol_target=0.5)
    from types import MappingProxyType

    preset_cfg = {
        "vol_adjust": MappingProxyType(
            {
                "window": MappingProxyType({"length": 10}),
                "enabled": None,
            }
        )
    }
    preset = presets.TrendPreset(
        slug="window",
        label="Window",
        description="",
        trend_spec=spec,
        _config=presets._freeze_mapping(preset_cfg),
    )
    defaults = preset.vol_adjust_defaults()
    assert defaults["enabled"] is True
    assert defaults["target_vol"] == 0.5
    # MappingProxyType should be copied into a mutable dictionary
    defaults["window"]["length"] = 12
    assert defaults["window"]["length"] == 12


def test_vol_adjust_defaults_uses_spec_when_no_signals() -> None:
    spec = TrendSpec(window=15, vol_adjust=True)
    preset = presets.TrendPreset(
        slug="spec",
        label="Spec",
        description="",
        trend_spec=spec,
        _config=presets._freeze_mapping({}),
    )
    defaults = preset.vol_adjust_defaults()
    assert defaults["enabled"] is True


@pytest.fixture()
def preset_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    base = tmp_path / "base"
    override = tmp_path / "override"
    env_dir = tmp_path / "env"
    for path in (base, override, env_dir):
        path.mkdir(parents=True)
    monkeypatch.setenv("TREND_PRESETS_DIR", str(env_dir))
    return base, override, env_dir


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_preset_registry_precedence_and_warning(
    preset_paths, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    base, override, env_dir = preset_paths
    _write_yaml(
        base / "alpha.yml",
        """
name: Alpha
lookback_months: 12
signals:
  window: 18
  lag: 2
""",
    )
    _write_yaml(
        override / "alpha.yml",
        """
name: Override Alpha
signals:
  window: 10
  min_periods: 5
metrics:
  sharpe: 1
""",
    )
    _write_yaml(
        env_dir / "beta.yml",
        """
name: Beta
metrics:
  sharpe: 0.5
  drawdown: 0.1
""",
    )

    monkeypatch.setattr(
        presets,
        "_candidate_preset_dirs",
        lambda: (base, override, env_dir),
    )
    caplog.set_level(logging.WARNING)
    _reset_registry()
    registry = presets._preset_registry()

    assert set(registry.keys()) == {"alpha", "beta"}
    # Environment override appended last; duplicates should emit a warning
    assert any(
        "Duplicate trend preset slug 'alpha'" in message for message in caplog.messages
    )
    alpha = registry["alpha"]
    assert alpha.trend_spec.window == 10
    beta = registry["beta"]
    assert isinstance(beta, presets.TrendPreset)

    # Cache should return identical mapping without re-reading files
    again = presets._preset_registry()
    assert again is registry


def test_listing_and_lookup_helpers(
    preset_paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, override, env_dir = preset_paths
    _write_yaml(
        base / "gamma.yml",
        """
name: Gamma
metrics:
  sharpe: 0.2
""",
    )
    _write_yaml(
        base / "labelled.yml",
        """
name: Friendly
description: Example preset
signals:
  window: 5
""",
    )
    monkeypatch.setattr(
        presets,
        "_candidate_preset_dirs",
        lambda: (base,),
    )
    _reset_registry()
    presets_list = presets.list_trend_presets()
    assert any(p.slug == "gamma" for p in presets_list)
    assert "gamma" in presets.list_preset_slugs()
    # Lookup by label (not slug) should work regardless of case
    preset = presets.get_trend_preset("GAMMA")
    assert preset.label == "Gamma"
    # Requesting by the exact label exercises the label-matching branch
    preset = presets.get_trend_preset("Friendly")
    assert preset.slug == "labelled"
    with pytest.raises(KeyError):
        presets.get_trend_preset("missing")
    with pytest.raises(KeyError):
        presets.get_trend_preset("")


def test_yaml_loader_and_metric_aliases(tmp_path: Path) -> None:
    file_path = tmp_path / "invalid.yml"
    _write_yaml(file_path, "- not a mapping")
    data = presets._load_yaml(file_path)
    assert data == {}

    assert presets.normalise_metric_key("Max_Drawdown") == "drawdown"
    assert presets.normalise_metric_key("") is None
    assert presets.pipeline_metric_key("vol") == "Volatility"
    assert presets.pipeline_metric_key("unknown") is None
    assert presets.pipeline_metric_key("") is None


def test_apply_trend_preset_merges_existing_signal_mapping() -> None:
    spec = TrendSpec(window=30, vol_adjust=True, vol_target=0.3)
    preset = presets.TrendPreset(
        slug="dict",
        label="Dict",
        description="",
        trend_spec=spec,
        _config=presets._freeze_mapping({"vol_adjust": {"enabled": True}}),
    )
    from types import MappingProxyType

    cfg = SimpleNamespace(
        signals={"window": 10},
        vol_adjust=MappingProxyType({"enabled": False}),
        run=MappingProxyType({}),
    )
    presets.apply_trend_preset(cfg, preset)
    assert cfg.signals["window"] == 30
    assert cfg.vol_adjust["enabled"] is True
    assert cfg.run["trend_preset"] == "dict"


def test_apply_trend_preset_handles_missing_signals() -> None:
    spec = TrendSpec(window=25)
    preset = presets.TrendPreset(
        slug="missing",
        label="Missing",
        description="",
        trend_spec=spec,
        _config=presets._freeze_mapping({}),
    )
    cfg = SimpleNamespace(signals=None, vol_adjust="skip", run=None)
    presets.apply_trend_preset(cfg, preset)
    assert cfg.signals["window"] == 25
    assert cfg.vol_adjust["enabled"] is True
    assert cfg.run["trend_preset"] == "missing"


def test_candidate_dirs_includes_env_and_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    default_dir = repo_root / "config" / "presets"
    default_dir.mkdir(parents=True)
    module_dir = repo_root / "src" / "trend_analysis"
    module_dir.mkdir(parents=True)
    fake_file = module_dir / "presets.py"
    fake_file.write_text("", encoding="utf-8")

    extra_parent_dir = repo_root / "src" / "config" / "presets"
    extra_parent_dir.mkdir(parents=True)

    monkeypatch.setattr(presets, "__file__", str(fake_file))
    monkeypatch.setattr(presets, "PRESETS_DIR", default_dir)
    monkeypatch.setattr(presets, "_DEFAULT_PRESETS_DIR", default_dir)
    # Register the same path twice to exercise the de-duplication branch
    monkeypatch.setenv("TREND_PRESETS_DIR", str(default_dir))

    candidates = presets._candidate_preset_dirs()
    assert default_dir in candidates
    assert extra_parent_dir in candidates
    assert candidates.count(default_dir) == 1


def test_preset_registry_skips_empty_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "presets"
    directory.mkdir()
    _write_yaml(directory / "empty.yml", "[]")
    _write_yaml(
        directory / "delta.yml",
        """
name: Delta
signals:
  window: 5
""",
    )
    monkeypatch.setattr(presets, "_candidate_preset_dirs", lambda: (directory,))
    _reset_registry()
    registry = presets._preset_registry()
    assert set(registry) == {"delta"}
