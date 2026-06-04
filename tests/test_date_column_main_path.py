from types import SimpleNamespace

import pandas as pd
import pytest

from trend_analysis import pipeline_entrypoints, run_analysis
from trend_analysis.data import load_csv
from trend_analysis.io.market_data import MarketDataValidationError
from trend_analysis.multi_period import engine as multi_period_engine
from trend_analysis.multi_period import loaders as multi_period_loaders


class DummyResult(SimpleNamespace):
    metrics: pd.DataFrame
    details: dict


def test_load_csv_honors_configured_date_column(tmp_path):
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text(
        "Timestamp,ManagerA,ManagerB\n"
        "2024-01-31,0.01,0.02\n"
        "2024-02-29,0.03,0.04\n"
    )

    frame = load_csv(str(csv_path), errors="raise", date_column="Timestamp")

    assert frame is not None
    assert list(frame.columns) == ["Date", "ManagerA", "ManagerB"]
    assert pd.api.types.is_datetime64_any_dtype(frame["Date"])


def test_load_csv_still_accepts_standard_date_column(tmp_path):
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("Date,ManagerA\n2024-01-31,0.01\n2024-02-29,0.03\n")

    frame = load_csv(str(csv_path), errors="raise")

    assert frame is not None
    assert list(frame.columns) == ["Date", "ManagerA"]


def test_load_csv_rejects_missing_configured_date_column(tmp_path):
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text(
        "Date,ManagerA\n"
        "2024-01-31,0.01\n"
        "2024-02-29,0.03\n"
    )

    with pytest.raises(MarketDataValidationError, match="Timestamp"):
        load_csv(str(csv_path), errors="raise", date_column="Timestamp")


def test_accepts_keyword_is_conservative_when_signature_fails():
    marker = object()

    assert pipeline_entrypoints._accepts_keyword(marker, "date_column") is False
    assert multi_period_loaders._accepts_keyword(marker, "date_column") is False
    assert multi_period_engine._accepts_keyword(marker, "date_column") is False


def test_run_analysis_passes_configured_date_column(monkeypatch, tmp_path):
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("Timestamp,ManagerA\n2024-01-31,0.01\n")
    cfg = SimpleNamespace(
        data={"csv_path": str(csv_path), "date_column": "Timestamp"},
        sample_split={
            "in_start": "2024-01-01",
            "in_end": "2024-01-31",
            "out_start": "2024-02-01",
            "out_end": "2024-02-29",
        },
        export={"directory": str(tmp_path), "formats": ["json"], "filename": "report"},
    )

    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)
    captured: dict[str, object] = {}

    def fake_load_csv(path, *, errors="raise", date_column="Date", **kwargs):
        captured.update(
            {
                "path": path,
                "errors": errors,
                "date_column": date_column,
                "kwargs": kwargs,
            }
        )
        return pd.DataFrame({"Date": pd.to_datetime(["2024-01-31"]), "ManagerA": [0.01]})

    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)
    monkeypatch.setattr(
        run_analysis.api,
        "run_simulation",
        lambda config, df: DummyResult(metrics=pd.DataFrame(), details={}),
    )

    assert run_analysis.main(["--config", "config.yml"]) == 0
    assert captured["path"] == str(csv_path)
    assert captured["errors"] == "raise"
    assert captured["date_column"] == "Timestamp"


def test_run_analysis_without_date_column_keeps_default(monkeypatch, tmp_path):
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("Date,ManagerA\n2024-01-31,0.01\n")
    cfg = SimpleNamespace(
        data={"csv_path": str(csv_path)},
        sample_split={
            "in_start": "2024-01-01",
            "in_end": "2024-01-31",
            "out_start": "2024-02-01",
            "out_end": "2024-02-29",
        },
        export={"directory": str(tmp_path), "formats": ["json"], "filename": "report"},
    )

    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)
    captured: dict[str, object] = {}

    def fake_load_csv(path, *, errors="raise", date_column="Date", **kwargs):
        captured["date_column"] = date_column
        return pd.DataFrame({"Date": pd.to_datetime(["2024-01-31"]), "ManagerA": [0.01]})

    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)
    monkeypatch.setattr(
        run_analysis.api,
        "run_simulation",
        lambda config, df: DummyResult(metrics=pd.DataFrame(), details={}),
    )

    assert run_analysis.main(["--config", "config.yml"]) == 0
    assert captured["date_column"] == "Date"
