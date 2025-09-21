"""Additional tests for ``trend_analysis.config.models`` coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

import yaml  # type: ignore[import-untyped]
from pathlib import Path

import pytest

from trend_analysis.config import models


def _base_config() -> dict[str, object]:
    """Return a minimal configuration mapping accepted by ``models.Config``."""

    return {
        "version": "1.0",
        "data": {
            "managers_glob": "data/raw/managers/*.csv",
            "date_column": "Date",
            "frequency": "D",
        },
        "preprocessing": {},
        "vol_adjust": {"target_vol": 0.1},
        "sample_split": {},
        "portfolio": {
            "selection_mode": "all",
            "rebalance_calendar": "NYSE",
            "max_turnover": 1.0,
            "transaction_cost_bps": 0,
        },
        "benchmarks": {},
        "metrics": {},
        "export": {},
        "run": {},
    }


def test_column_mapping_defaults_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the ``ColumnMapping`` defaults and validation branches."""

    with pytest.raises(ValueError, match="Date column must be specified"):
        models.ColumnMapping(return_columns=["ret"])

    with pytest.raises(
        ValueError, match="At least one return column must be specified"
    ):
        models.ColumnMapping(date_column="Date")

    mapping = models.ColumnMapping(
        date_column="Date",
        return_columns=["ret"],
        column_display_names=None,
        column_tickers=None,
    )

    assert mapping.column_display_names == {}
    assert mapping.column_tickers == {}


def test_load_merges_output_section(tmp_path: Path) -> None:
    """``load`` should fold ``output`` metadata into the export settings."""

    export_target = tmp_path / "exports" / "report.xlsx"
    config_dict = _base_config()
    config_dict["export"] = {"formats": ("json",)}
    config_dict["output"] = {
        "format": ["CSV", "json", "csv"],
        "path": str(export_target),
    }

    config = models.load(config_dict)

    assert config.export["formats"] == ["json", "CSV"]
    assert config.export["directory"] == str(export_target.parent)
    assert config.export["filename"] == export_target.name


def test_load_uses_environment_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no path is provided, ``load`` should honour ``TREND_CFG``."""

    config_path = tmp_path / "custom.yml"
    payload = _base_config()
    payload["version"] = "from-env"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    monkeypatch.setenv("TREND_CFG", str(config_path))
    try:
        config = models.load()
    finally:
        monkeypatch.delenv("TREND_CFG", raising=False)

    assert config.version == "from-env"


def test_load_config_rejects_invalid_type() -> None:
    """``load_config`` only accepts mappings or path-like objects."""

    with pytest.raises(TypeError, match="cfg must be a mapping or path"):
        models.load_config(123)  # type: ignore[arg-type]


def test_list_available_presets_handles_missing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing config directory should return an empty preset list."""

    missing = tmp_path / "not_there"
    monkeypatch.setattr(models, "_find_config_directory", lambda: missing)

    assert models.list_available_presets() == []


def test_preset_listing_and_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Presets are discovered and loaded via ``_find_config_directory``."""

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "defaults.yml").write_text("version: default\n", encoding="utf-8")

    preset_payload = {
        "description": "demo",
        "data": {
            "managers_glob": "data/raw/managers/*.csv",
            "date_column": "Date",
            "frequency": "D",
        },
        "preprocessing": {},
        "vol_adjust": {"target_vol": 0.1},
        "sample_split": {},
        "portfolio": {
            "selection_mode": "all",
            "rebalance_calendar": "NYSE",
            "max_turnover": 1.0,
            "transaction_cost_bps": 0,
        },
        "metrics": {},
        "export": {},
        "run": {},
    }
    (config_dir / "alpha.yml").write_text(
        yaml.safe_dump(preset_payload), encoding="utf-8"
    )
    (config_dir / "beta.yml").write_text("[]\n", encoding="utf-8")

    monkeypatch.setattr(models, "_find_config_directory", lambda: config_dir)

    assert models.list_available_presets() == ["alpha", "beta"]

    preset = models.load_preset("alpha")
    assert preset.name == "alpha"
    assert preset.description == "demo"

    with pytest.raises(TypeError, match="Preset file must contain a mapping"):
        models.load_preset("beta")


def test_load_raises_for_non_mapping_file(tmp_path: Path) -> None:
    """``load`` should reject YAML files that do not contain mappings."""

    cfg_file = tmp_path / "bad.yml"
    cfg_file.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(TypeError, match="Config file must contain a mapping"):
        models.load(cfg_file)
