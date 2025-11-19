from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trend_analysis.multi_period import loaders


@dataclass
class LoaderConfig:
    data: dict[str, Any] = field(default_factory=dict)
    benchmarks: dict[str, str] = field(default_factory=dict)


def test_load_prices_reads_csv(tmp_path: Path) -> None:
    csv = tmp_path / "returns.csv"
    csv.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    cfg = LoaderConfig(data={"csv_path": csv, "nan_policy": "ffill"})
    frame = loaders.load_prices(cfg)
    assert list(frame.columns) == ["Date", "A"]
    assert frame.iloc[0]["A"] == pytest.approx(0.1)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.tz is timezone.utc
    assert str(frame["Date"].dtype) == "datetime64[ns, UTC]"


def test_load_prices_requires_path() -> None:
    cfg = LoaderConfig()
    with pytest.raises(KeyError):
        loaders.load_prices(cfg)


def test_load_prices_passes_frequency_to_contract(monkeypatch, tmp_path: Path) -> None:
    csv = tmp_path / "returns.csv"
    csv.write_text("Date,A\n2020-01-31,0.1\n2020-02-29,0.2\n", encoding="utf-8")
    cfg = LoaderConfig(data={"csv_path": csv})

    captured: dict[str, Any] = {}

    def fake_validate(frame: pd.DataFrame, *, freq: str | None = None) -> pd.DataFrame:
        captured["freq"] = freq
        captured["rows"] = len(frame)
        return frame

    monkeypatch.setattr(loaders, "validate_prices", fake_validate)

    frame = loaders.load_prices(cfg)
    assert captured["rows"] == len(frame)
    expected_freq = frame.attrs.get("market_data_frequency_code") or "D"
    assert captured["freq"] == expected_freq


def test_load_membership_normalises_schema(tmp_path: Path) -> None:
    ledger = tmp_path / "membership.csv"
    ledger.write_text(
        "Fund,Effective_Date,End_Date\nFundA,2020-01-31,\n",
        encoding="utf-8",
    )
    cfg = LoaderConfig(data={"universe_membership_path": ledger})
    frame = loaders.load_membership(cfg)
    assert list(frame.columns) == ["fund", "effective_date", "end_date"]
    assert frame.iloc[0]["fund"] == "FundA"
    assert str(frame.iloc[0]["effective_date"]) == "2020-01-31 00:00:00"


def test_load_membership_empty_when_missing_path() -> None:
    cfg = LoaderConfig()
    frame = loaders.load_membership(cfg)
    assert frame.empty
    assert list(frame.columns) == ["fund", "effective_date", "end_date"]


def test_load_benchmarks_extracts_columns() -> None:
    prices = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "SPX": [0.1, 0.2],
        }
    )
    cfg = LoaderConfig(benchmarks={"spx": "SPX"})
    bench = loaders.load_benchmarks(cfg, prices)
    assert list(bench.columns) == ["Date", "spx"]
    assert bench.iloc[1]["spx"] == pytest.approx(0.2)


def test_load_benchmarks_raises_for_missing_column() -> None:
    prices = pd.DataFrame({"Date": pd.to_datetime(["2020-01-31"])})
    cfg = LoaderConfig(benchmarks={"spx": "SPX"})
    with pytest.raises(KeyError):
        loaders.load_benchmarks(cfg, prices)
