from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from trend_analysis import tool_layer
from trend_analysis.tool_layer import ToolLayer


def _sample_config() -> dict[str, Any]:
    return {
        "version": "1",
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


def test_run_analysis_uses_passed_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = ToolLayer()
    cfg = _sample_config()
    df = pd.DataFrame({"Date": ["2020-01-01"], "A": [0.01]})
    captured: dict[str, Any] = {}

    def fake_run_simulation(config, frame):
        captured["config"] = config
        captured["frame"] = frame
        return "ok"

    monkeypatch.setattr(tool_layer.api, "run_simulation", fake_run_simulation)

    result = tool.run_analysis(cfg, data=df)

    assert result.success is True
    assert result.data == "ok"
    assert captured["frame"] is df


def test_run_analysis_loads_csv_when_data_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = ToolLayer()
    cfg = _sample_config()
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("Date,A\n2020-01-01,0.01\n", encoding="utf-8")
    cfg["data"] = {
        **cfg["data"],
        "csv_path": str(csv_path),
        "missing_policy": "ffill",
        "missing_limit": 2,
    }
    df = pd.DataFrame({"Date": ["2020-01-01"], "A": [0.01]})
    captured: dict[str, Any] = {}

    def fake_load_csv(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return df

    monkeypatch.setattr(tool_layer, "load_csv", fake_load_csv)
    monkeypatch.setattr(tool_layer, "api", SimpleNamespace(run_simulation=lambda *_: "ok"))

    result = tool.run_analysis(cfg)

    assert result.success is True
    assert result.data == "ok"
    assert captured["path"] == str(csv_path)
    assert captured["kwargs"]["errors"] == "raise"
    assert captured["kwargs"]["missing_policy"] == "ffill"
    assert captured["kwargs"]["missing_limit"] == 2


def test_run_analysis_requires_csv_path() -> None:
    tool = ToolLayer()

    result = tool.run_analysis(_sample_config())

    assert result.success is False
    assert "csv_path" in (result.error or "")


def test_run_analysis_rejects_non_dataframe_data() -> None:
    tool = ToolLayer()

    result = tool.run_analysis(_sample_config(), data="not-a-dataframe")

    assert result.success is False
    assert "pandas DataFrame" in (result.error or "")
