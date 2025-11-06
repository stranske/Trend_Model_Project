"""Additional coverage for ``trend_analysis.io.market_data`` helpers."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

import trend_analysis.io.market_data as market_data


@pytest.fixture()
def base_metadata_kwargs() -> dict[str, object]:
    """Provide the minimal arguments required to build metadata instances."""

    return {
        "mode": market_data.MarketDataMode.RETURNS,
        "frequency": "M",
        "frequency_label": "monthly",
        "start": datetime(2024, 1, 31),
        "end": datetime(2024, 4, 30),
        "rows": 4,
    }


def test_metadata_syncs_columns_and_symbols_and_date_range(
    base_metadata_kwargs: dict[str, object],
) -> None:
    metadata_with_columns = market_data.MarketDataMetadata(
        columns=["FundA", "FundB"],
        **base_metadata_kwargs,
    )
    # Symbols populate automatically from columns when omitted.
    assert metadata_with_columns.symbols == ["FundA", "FundB"]
    assert metadata_with_columns.date_range == ("2024-01-31", "2024-04-30")

    metadata_with_symbols = market_data.MarketDataMetadata(
        symbols=["Alpha"],
        **base_metadata_kwargs,
    )
    # Columns mirror symbols when the original input omits them.
    assert metadata_with_symbols.columns == ["Alpha"]


def test_validated_market_data_delegates_dataframe_behaviour(
    base_metadata_kwargs: dict[str, object],
) -> None:
    frame = pd.DataFrame(
        {"FundA": [0.1, 0.2]},
        index=pd.date_range("2024-01-31", periods=2, freq="M"),
    )
    metadata = market_data.MarketDataMetadata(columns=["FundA"], **base_metadata_kwargs)
    validated = market_data.ValidatedMarketData(frame=frame, metadata=metadata)

    # ``__iter__`` exposes the DataFrame columns and ``to_frame`` returns
    # the original payload.
    assert list(iter(validated)) == ["FundA"]
    assert validated.to_frame().equals(frame)


def test_apply_missing_policy_ffill_drops_all_nan_columns() -> None:
    frame = pd.DataFrame({"A": [pd.NA, pd.NA, pd.NA]})
    result, summary = market_data.apply_missing_policy(frame, policy="ffill")
    assert result.empty
    assert summary["dropped"] == ["A"]


def test_summarise_missing_policy_handles_mixed_detail_types() -> None:
    info = {
        "policy": "ffill",
        "limit": 2,
        "policy_map": {"A": "ffill", "B": "zero"},
        "filled": {
            "A": market_data.MissingPolicyFillDetails(method="ffill", count=3),
            "B": {"method": "zero", "count": "invalid"},
            "C": object(),
        },
        "dropped": ["D"],
    }

    summary = market_data._summarise_missing_policy(info)

    assert "policy=ffill" in summary
    # The overrides section should be present because column ``B`` differs
    # from the default policy.
    assert "overrides=B:zero" in summary
    # The summary surfaces each filled column, including those that required
    # type coercion or default fall-backs.
    assert "filled=A (ffill: 3)" in summary
    assert "B (zero: 0)" in summary
    assert "C (fill: 0)" in summary
    assert "dropped=D" in summary


def test_classify_frequency_handles_short_and_zero_offsets() -> None:
    single = market_data.classify_frequency(pd.DatetimeIndex(["2024-01-31"]))
    assert single["code"] == "UNKNOWN"

    duplicates = market_data.classify_frequency(pd.DatetimeIndex(["2024-01-31", pd.NaT]))
    assert duplicates["code"] == "UNKNOWN"

    with pytest.raises(market_data.MarketDataValidationError):
        market_data.classify_frequency(
            pd.DatetimeIndex(
                [
                    "2024-01-31",
                    "2024-01-31",
                    "2024-01-31",
                    "2024-02-29",
                ]
            )
        )


def test_resolve_datetime_index_reports_unparseable_values() -> None:
    df = pd.DataFrame(
        {
            "Date": ["2024-01-31", "not-a-date"],
            "FundA": [0.1, 0.2],
        }
    )

    with pytest.raises(market_data.MarketDataValidationError) as exc:
        market_data._resolve_datetime_index(df, source="upload.csv")

    assert "could not be parsed" in str(exc.value)


def test_resolve_datetime_index_handles_parser_exception(monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "Date": ["bad", "values"],
            "FundA": [0.1, 0.2],
        }
    )

    def raise_value_error(*args: object, **kwargs: object) -> pd.Series:
        raise ValueError("boom")

    monkeypatch.setattr(market_data.pd, "to_datetime", raise_value_error)

    with pytest.raises(market_data.MarketDataValidationError) as exc:
        market_data._resolve_datetime_index(df, source="upload.csv")

    assert "Found dates that could not be parsed" in str(exc.value)


def test_resolve_datetime_index_without_data_columns() -> None:
    df = pd.DataFrame({"Date": ["2024-01-31", "2024-02-29"]})

    with pytest.raises(market_data.MarketDataValidationError) as exc:
        market_data._resolve_datetime_index(df, source="upload.csv")

    assert "No data columns" in str(exc.value)


def test_check_monotonic_index_reports_out_of_order() -> None:
    index = pd.DatetimeIndex(["2024-01-03", "2024-01-01", "2024-01-02"])
    issues = market_data._check_monotonic_index(index)

    assert any("out-of-order" in issue for issue in issues)


def test_check_monotonic_index_reports_many_duplicates() -> None:
    repeated: list[str] = []
    for month in range(1, 8):
        stamp = f"2024-0{month}-15"
        repeated.extend([stamp, stamp])

    index = pd.to_datetime(repeated)
    issues = market_data._check_monotonic_index(index)

    assert any("Duplicate timestamps" in issue for issue in issues)
    assert "…" in issues[0]


def test_column_mode_returns_none_for_empty_series() -> None:
    series = pd.Series([float("nan"), float("nan")], dtype="float64")
    assert market_data._column_mode(series) is None


def test_infer_mode_reports_ambiguous_columns() -> None:
    df = pd.DataFrame(
        {
            "Returns": [0.01, -0.02, 0.03],
            "Hybrid": [-50.0, 45.0, -55.0],
        }
    )

    with pytest.raises(market_data.MarketDataValidationError) as exc:
        market_data._infer_mode(df)

    assert "Could not classify columns" in str(exc.value)


def test_load_market_data_csv_success(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_read_csv(path: str) -> pd.DataFrame:
        captured["path"] = path
        return pd.DataFrame({"Date": ["2024-01-31"], "FundA": [0.1]})

    def fake_validate(frame: pd.DataFrame, *, source: str | None = None):
        captured["source"] = source
        captured["frame_columns"] = list(frame.columns)
        return SimpleNamespace(result="ok")

    monkeypatch.setattr(market_data.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(market_data, "validate_market_data", fake_validate)

    result = market_data.load_market_data_csv("/tmp/data.csv")

    assert result.result == "ok"
    assert captured == {
        "path": "/tmp/data.csv",
        "source": "/tmp/data.csv",
        "frame_columns": ["Date", "FundA"],
    }


def test_load_market_data_csv_error_paths(monkeypatch) -> None:
    def raise_empty(*args: object, **kwargs: object) -> pd.DataFrame:
        raise pd.errors.EmptyDataError("empty")

    monkeypatch.setattr(market_data.pd, "read_csv", raise_empty)

    with pytest.raises(market_data.MarketDataValidationError) as exc:
        market_data.load_market_data_csv("file.csv")
    assert "File contains no data" in str(exc.value)

    def raise_parser(*args: object, **kwargs: object) -> pd.DataFrame:
        raise pd.errors.ParserError("parse", "details")

    monkeypatch.setattr(market_data.pd, "read_csv", raise_parser)

    with pytest.raises(market_data.MarketDataValidationError) as exc:
        market_data.load_market_data_csv("file.csv")
    assert "Failed to parse file" in str(exc.value)


def test_load_market_data_parquet_success(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_read_parquet(path: str) -> pd.DataFrame:
        captured["path"] = path
        return pd.DataFrame({"Date": ["2024-01-31"], "FundA": [0.1]})

    def fake_validate(frame: pd.DataFrame, *, source: str | None = None):
        captured["source"] = source
        return "validated"

    monkeypatch.setattr(market_data.pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(market_data, "validate_market_data", fake_validate)

    result = market_data.load_market_data_parquet("/tmp/data.parquet")

    assert result == "validated"
    assert captured == {"path": "/tmp/data.parquet", "source": "/tmp/data.parquet"}


def test_load_market_data_parquet_permission_error(monkeypatch) -> None:
    def raise_permission(*args: object, **kwargs: object) -> pd.DataFrame:
        raise PermissionError("denied")

    monkeypatch.setattr(market_data.pd, "read_parquet", raise_permission)

    with pytest.raises(market_data.MarketDataValidationError) as exc:
        market_data.load_market_data_parquet("/tmp/data.parquet")

    assert "Permission denied" in str(exc.value)
