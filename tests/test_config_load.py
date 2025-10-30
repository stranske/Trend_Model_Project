from __future__ import annotations

from pathlib import Path

import yaml

from trend_analysis import config


def _write_cfg(
    path: Path,
    version: str,
    *,
    csv_path: Path,
    extra: dict[str, object] | None = None,
) -> None:
    try:
        csv_value = str(csv_path.relative_to(path.parent))
    except ValueError:
        csv_value = str(csv_path)

    payload: dict[str, object] = {
        "version": version,
        "data": {
            "csv_path": csv_value,
            "date_column": "Date",
            "frequency": "M",
        },
        "preprocessing": {},
        "vol_adjust": {"target_vol": 0.15},
        "sample_split": {},
        "portfolio": {
            "selection_mode": "all",
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.25,
            "transaction_cost_bps": 10,
        },
        "metrics": {},
        "export": {},
        "run": {},
    }

    if extra:
        payload.update(extra)

    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _make_csv(path: Path) -> Path:
    path.write_text("Date,RF\n2020-01-31,0.0\n", encoding="utf-8")
    return path


def test_load_default():
    cfg = config.load()
    assert isinstance(cfg, config.Config)
    assert cfg.version


def test_load_custom(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yml"
    csv_path = _make_csv(tmp_path / "data.csv")
    _write_cfg(cfg_path, "99", csv_path=csv_path)
    cfg = config.load(str(cfg_path))
    assert cfg.version == "99"


def test_env_var_override(tmp_path: Path, monkeypatch) -> None:
    cfg_file = tmp_path / "env.yml"
    csv_path = _make_csv(tmp_path / "data.csv")
    _write_cfg(cfg_file, "42", csv_path=csv_path)
    monkeypatch.setenv("TREND_CFG", str(cfg_file))
    cfg = config.load()
    assert cfg.version == "42"
    monkeypatch.delenv("TREND_CFG", raising=False)


def test_output_alias(tmp_path: Path) -> None:
    cfg_file = tmp_path / "alias.yml"
    csv_path = _make_csv(tmp_path / "data.csv")
    _write_cfg(
        cfg_file,
        "1",
        csv_path=csv_path,
        extra={
            "output": {"format": "csv", "path": str(tmp_path / "res")},
        },
    )
    cfg = config.load(str(cfg_file))
    assert cfg.export["formats"] == ["csv"]
    assert cfg.export["directory"] == str(tmp_path)
    assert cfg.export["filename"] == "res"


def test_empty_version_rejected(tmp_path: Path) -> None:
    cfg_file = tmp_path / "empty_version.yml"
    csv_path = _make_csv(tmp_path / "data.csv")
    _write_cfg(cfg_file, "", csv_path=csv_path)
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        config.load(str(cfg_file))
    assert "String should have at least 1 character" in str(exc_info.value)


def test_whitespace_version_rejected(tmp_path: Path) -> None:
    cfg_file = tmp_path / "whitespace_version.yml"
    csv_path = _make_csv(tmp_path / "data.csv")
    _write_cfg(cfg_file, "   ", csv_path=csv_path)
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        config.load(str(cfg_file))
    assert "Version field cannot be empty" in str(exc_info.value)
