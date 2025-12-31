import json
from pathlib import Path

import pandas as pd
import pytest

from trend_analysis.walk_forward import (
    DataConfig,
    RunConfig,
    StrategyConfig,
    WalkForwardSettings,
    WindowConfig,
    infer_periods_per_year,
    load_returns,
    load_settings,
    persist_artifacts,
)


def _write_config(tmp_path: Path, content: dict) -> Path:
    cfg_path = tmp_path / "wf.json"
    cfg_path.write_text(json.dumps(content), encoding="utf-8")
    return cfg_path


def test_load_settings_resolves_relative_paths(tmp_path: Path) -> None:
    data_csv = tmp_path / "data.csv"
    data_csv.write_text("Date,A,B\n2020-01-01,0.1,0.2", encoding="utf-8")

    cfg = {
        "data": {"csv_path": "data.csv", "date_column": "Date", "columns": ["A", "B"]},
        "walk_forward": {"train": 2, "test": 1, "step": 1},
        "strategy": {
            "top_n": 3,
            "defaults": {"band": -1},
            "grid": {"lookback": [2, 3]},
        },
        "run": {"name": "demo", "output_dir": "out", "seed": 7},
    }
    cfg_path = _write_config(tmp_path, cfg)

    settings = load_settings(cfg_path)

    assert isinstance(settings, WalkForwardSettings)
    assert settings.data.csv_path == data_csv.resolve()
    assert settings.windows == WindowConfig(train=2, test=1, step=1)
    assert settings.strategy.grid == {"lookback": [2, 3]}
    assert settings.run.output_dir == (tmp_path / "out").resolve()
    assert settings.run.seed == 7


def test_load_settings_rejects_missing_csv_path(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path,
        {
            "data": {},
            "walk_forward": {"train": 1, "test": 1, "step": 1},
            "strategy": {"grid": {"lookback": [1]}},
            "run": {},
        },
    )

    with pytest.raises(ValueError, match="data.csv_path must be provided"):
        load_settings(cfg_path)


def test_load_settings_accepts_absolute_paths(tmp_path: Path) -> None:
    data_csv = tmp_path / "data.csv"
    data_csv.write_text("Date,A\n2020-01-01,0.1", encoding="utf-8")
    output_dir = tmp_path / "absolute_output"
    cfg_path = _write_config(
        tmp_path,
        {
            "data": {"csv_path": str(data_csv)},
            "walk_forward": {"train": 1, "test": 1, "step": 1},
            "strategy": {"grid": {"lookback": [1]}},
            "run": {"output_dir": str(output_dir)},
        },
    )

    settings = load_settings(cfg_path)

    assert settings.data.csv_path == data_csv
    assert settings.run.output_dir == output_dir


@pytest.mark.parametrize(
    "cfg, message",
    [
        (
            {
                "data": {"csv_path": "data.csv"},
                "walk_forward": {"train": 1, "test": 1, "step": 1},
                "strategy": {"grid": "not-a-mapping"},
                "run": {},
            },
            "strategy.grid must contain at least one parameter list",
        ),
        (
            {
                "data": {"csv_path": "data.csv", "columns": "A"},
                "walk_forward": {"train": 1, "test": 1, "step": 1},
                "strategy": {"grid": {"lookback": [1]}},
                "run": {},
            },
            "data.columns",
        ),
        (
            {
                "data": {"csv_path": "data.csv"},
                "walk_forward": {"train": 1, "test": 1, "step": 1},
                "strategy": {"defaults": "nope", "grid": {"lookback": [1]}},
                "run": {},
            },
            "strategy.defaults",
        ),
        (
            {
                "data": {"csv_path": "data.csv"},
                "walk_forward": {"train": 0, "test": 1, "step": 1},
                "strategy": {"grid": {"lookback": [1]}},
                "run": {},
            },
            "walk_forward.train",
        ),
        (
            {
                "data": {"csv_path": "data.csv"},
                "walk_forward": {"train": 1, "test": 1, "step": 1},
                "strategy": {"top_n": 0, "grid": {"lookback": [1]}},
                "run": {},
            },
            "strategy.top_n",
        ),
        (
            {
                "data": {"csv_path": "data.csv"},
                "walk_forward": {"train": 1, "test": 1, "step": 1},
                "strategy": {"grid": {"lookback": []}},
                "run": {},
            },
            "strategy.grid entry",
        ),
        (
            {
                "data": {"csv_path": "data.csv"},
                "walk_forward": {"train": 1, "test": 1, "step": 1},
                "strategy": {"grid": {"lookback": 2}},
                "run": {},
            },
            "strategy.grid values",
        ),
    ],
)
def test_load_settings_validation_errors(
    tmp_path: Path, cfg: dict, message: str
) -> None:
    cfg_path = _write_config(tmp_path, cfg)
    with pytest.raises(ValueError) as err:
        load_settings(cfg_path)
    assert message in str(err.value)


def test_load_returns_filters_columns_and_validates(tmp_path: Path) -> None:
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text(
        "Date,A,B,C\n2020-01-01,0.1,0.2,foo\n2020-02-01,0.3,0.4,bar",
        encoding="utf-8",
    )
    cfg = DataConfig(csv_path=csv_path, date_column="Date", columns=["B", "A"])

    df = load_returns(cfg)

    assert list(df.columns) == ["B", "A"]
    assert list(df.index) == sorted(df.index)

    bad_cfg = DataConfig(csv_path=csv_path, date_column="NotAColumn")
    with pytest.raises(ValueError, match="Date column 'NotAColumn'"):
        load_returns(bad_cfg)

    empty_cfg = DataConfig(csv_path=tmp_path / "empty.csv", date_column="Date")
    empty_cfg.csv_path.write_text("Date\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No numeric columns"):
        load_returns(empty_cfg)


def test_infer_periods_per_year_handles_varied_frequencies() -> None:
    idx_daily = pd.date_range("2020-01-01", periods=5, freq="D")
    idx_weekly = pd.date_range("2020-01-03", periods=4, freq="W-FRI")
    idx_monthly = pd.date_range("2020-01-31", periods=3, freq="M")

    assert infer_periods_per_year(idx_daily) == 252
    assert infer_periods_per_year(idx_weekly) == 52
    assert infer_periods_per_year(idx_monthly) == 12
    assert infer_periods_per_year(pd.DatetimeIndex([])) == 1


def test_infer_periods_per_year_handles_quarterly_frequency() -> None:
    idx_quarterly = pd.date_range("2020-01-01", periods=4, freq="QS")

    assert infer_periods_per_year(idx_quarterly) == 4


def test_persist_artifacts_emits_files(tmp_path: Path) -> None:
    settings = WalkForwardSettings(
        data=DataConfig(csv_path=tmp_path / "returns.csv"),
        windows=WindowConfig(train=1, test=1, step=1),
        strategy=StrategyConfig(grid={"lookback": [1]}),
        run=RunConfig(name="wf", output_dir=tmp_path / "wf_out"),
    )

    folds = pd.DataFrame(
        {
            "fold": [1],
            "train_start": [pd.Timestamp("2020-01-01")],
            "param_lookback": [1],
        }
    )
    summary = pd.DataFrame({"param_lookback": [1], "mean_cagr": [0.01], "folds": [1]})

    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text("run: {}", encoding="utf-8")

    run_dir = persist_artifacts(settings, folds, summary, config_path=cfg_path)

    assert run_dir.exists()
    assert (run_dir / "folds.csv").is_file()
    assert (run_dir / "summary.csv").is_file()
    assert (run_dir / "summary.jsonl").is_file()
    assert (run_dir / "config_used.yml").read_text(encoding="utf-8") == "run: {}"
