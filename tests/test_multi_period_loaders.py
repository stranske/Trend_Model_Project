"""Tests for multi-period loader helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.multi_period import loaders


class DummyConfig:
    def __init__(self, *, data=None, benchmarks=None):
        self.data = data or {}
        self.benchmarks = benchmarks


def test_load_prices_invokes_validators(monkeypatch, tmp_path):
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text("Date,AAA\n2024-01-01,1.0\n")

    called = {}

    def fake_load_csv(path, *, errors, missing_policy, missing_limit):
        called["path"] = path
        called["errors"] = errors
        called["missing_policy"] = missing_policy
        called["missing_limit"] = missing_limit
        frame = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "AAA": [1.0]})
        frame.attrs["market_data_frequency_code"] = "D"
        return frame

    def fake_coerce(frame):
        called["coerced"] = True
        return frame

    def fake_validate(frame, *, freq):
        called.setdefault("validated", []).append((tuple(frame.columns), freq))

    monkeypatch.setattr(loaders, "load_csv", fake_load_csv)
    monkeypatch.setattr(loaders, "coerce_to_utc", fake_coerce)
    monkeypatch.setattr(loaders, "validate_prices", fake_validate)

    cfg = DummyConfig(data={"csv_path": str(csv_path), "missing_policy": "ffill"})
    frame = loaders.load_prices(cfg)

    assert list(frame.columns) == ["Date", "AAA"]
    assert called["path"] == str(csv_path)
    assert called["errors"] == "raise"
    assert called["missing_policy"] == "ffill"
    assert called["missing_limit"] is None
    assert called["coerced"] is True
    assert called["validated"] == [(("Date", "AAA"), "D")]


def test_load_prices_falls_back_to_nan_config(monkeypatch, tmp_path):
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text("Date,AAA\n2024-01-01,1.0\n")

    captured = {}

    def fake_load_csv(path, *, errors, missing_policy, missing_limit):
        captured["path"] = path
        captured["errors"] = errors
        captured["missing_policy"] = missing_policy
        captured["missing_limit"] = missing_limit
        return pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "AAA": [1.0]})

    def fake_coerce(frame):
        captured["coerced"] = True
        return frame

    def fake_validate(frame, *, freq):
        captured.setdefault("validated", []).append((tuple(frame.columns), freq))

    monkeypatch.setattr(loaders, "load_csv", fake_load_csv)
    monkeypatch.setattr(loaders, "coerce_to_utc", fake_coerce)
    monkeypatch.setattr(loaders, "validate_prices", fake_validate)

    cfg = DummyConfig(
        data={"csv_path": str(csv_path), "nan_policy": "bfill", "nan_limit": 2}
    )

    frame = loaders.load_prices(cfg)

    assert list(frame.columns) == ["Date", "AAA"]
    assert captured["missing_policy"] == "bfill"
    assert captured["missing_limit"] == 2
    assert captured["validated"] == [(("Date", "AAA"), "D")]


def test_load_prices_missing_file_raises(tmp_path):
    cfg = DummyConfig(data={"csv_path": tmp_path / "missing.csv"})

    with pytest.raises(FileNotFoundError):
        loaders.load_prices(cfg)


def test_load_prices_missing_path_raises_key_error():
    cfg = DummyConfig(data={})
    with pytest.raises(KeyError):
        loaders.load_prices(cfg)


def test_coerce_path_rejects_directories(tmp_path):
    directory = tmp_path / "data_dir"
    directory.mkdir()

    with pytest.raises(IsADirectoryError):
        loaders._coerce_path(directory, field="data.csv_path")

    with pytest.raises(KeyError):
        loaders._coerce_path("   ", field="data.csv_path")


def test_coerce_path_resolves_relative_and_missing(tmp_path, monkeypatch):
    data_file = tmp_path / "data.csv"
    data_file.write_text("Date,AAA\n2024-01-01,1.0\n")

    monkeypatch.chdir(tmp_path)

    resolved = loaders._coerce_path("./data.csv", field="data.csv_path")
    assert resolved == data_file.resolve()

    with pytest.raises(FileNotFoundError):
        loaders._coerce_path("./missing.csv", field="data.csv_path")


def test_load_membership_normalises_and_sorts(tmp_path):
    membership = tmp_path / "universe.csv"
    membership.write_text(
        "Symbol,effective_date,end_date\nB,2024-02-02,\nA,2024-01-01,2024-02-01\n"
    )

    cfg = DummyConfig(data={"universe_membership_path": membership})
    result = loaders.load_membership(cfg)

    assert list(result.columns) == ["fund", "effective_date", "end_date"]
    assert result["fund"].tolist() == ["A", "B"]
    assert pd.isna(result.loc[1, "end_date"])


def test_load_membership_missing_required_columns_raises(tmp_path):
    membership = tmp_path / "universe.csv"
    membership.write_text("Symbol,end_date\nA,\n")

    cfg = DummyConfig(data={"universe_membership_path": membership})
    with pytest.raises(ValueError):
        loaders.load_membership(cfg)


def test_load_membership_returns_empty_when_not_configured():
    empty = loaders.load_membership(DummyConfig(data={}))

    assert empty.empty
    assert list(empty.columns) == ["fund", "effective_date", "end_date"]


def test_load_benchmarks_selects_columns_and_errors_on_missing():
    prices = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "AAA": [1.0, 2.0],
            "BBB": [3.0, 4.0],
        }
    )
    cfg = DummyConfig(benchmarks={"Bench1": "AAA", "Bench2": "CCC"})

    with pytest.raises(KeyError):
        loaders.load_benchmarks(cfg, prices)

    cfg_valid = DummyConfig(benchmarks={"Bench1": "AAA", "Bench2": "BBB"})
    frame = loaders.load_benchmarks(cfg_valid, prices)

    assert list(frame.columns) == ["Date", "Bench1", "Bench2"]
    assert frame["Bench2"].tolist() == [3.0, 4.0]


def test_load_benchmarks_returns_empty_when_no_mapping():
    prices = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"])})
    cfg = DummyConfig(benchmarks=None)

    empty = loaders.load_benchmarks(cfg, prices)
    assert empty.empty
    assert list(empty.columns) == ["Date"]


def test_load_benchmarks_requires_date_column():
    prices = pd.DataFrame({"AAA": [1.0]})
    cfg = DummyConfig(benchmarks={"Bench1": "AAA"})

    with pytest.raises(KeyError):
        loaders.load_benchmarks(cfg, prices)
