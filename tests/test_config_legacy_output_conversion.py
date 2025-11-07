from __future__ import annotations

from pathlib import Path

import yaml

from trend_analysis.config import legacy


def test_load_promotes_output_section(tmp_path):
    target = tmp_path / "exports" / "report.csv"
    target.parent.mkdir()
    config_data = {
        "version": "1.0",
        "data": {},
        "preprocessing": {},
        "vol_adjust": {},
        "sample_split": {},
        "portfolio": {},
        "benchmarks": {},
        "signals": {},
        "performance": {},
        "metrics": {},
        "export": {"formats": ["json"]},
        "output": {"format": "csv", "path": str(target)},
        "run": {},
    }
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    loaded = legacy.load(cfg_path)

    assert isinstance(loaded, legacy.Config)
    assert loaded.export["formats"] == ["csv"]
    assert loaded.export["directory"].endswith("exports")
    assert loaded.export["filename"] == "report.csv"


def test_load_supports_iterable_formats(tmp_path):
    cfg_path = tmp_path / "config.yml"
    config_data = {
        "version": "1.0",
        "data": {},
        "preprocessing": {},
        "vol_adjust": {},
        "sample_split": {},
        "portfolio": {},
        "benchmarks": {},
        "signals": {},
        "performance": {},
        "metrics": {},
        "export": {},
        "output": {"format": ("csv", "xlsx"), "path": "report.out"},
        "run": {},
    }
    cfg_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    loaded = legacy.load(cfg_path)

    assert loaded.export["formats"] == ["csv", "xlsx"]
    assert Path(loaded.export["directory"]) == Path(".")
    assert loaded.export["filename"] == "report.out"
