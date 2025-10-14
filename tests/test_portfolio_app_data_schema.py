"""Coverage-focused tests for ``trend_portfolio_app.data_schema``."""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest


@pytest.fixture
def schema_module() -> Any:
    import trend_portfolio_app.data_schema as data_schema

    return data_schema


def _dummy_validated(frame: pd.DataFrame) -> SimpleNamespace:
    metadata = SimpleNamespace(
        rows=len(frame),
        columns=list(frame.columns),
        symbols=["A", "B"],
        mode=SimpleNamespace(value="returns"),
        frequency_label="Monthly",
        frequency="M",
        frequency_detected="M",
        frequency_missing_periods=2,
        frequency_max_gap_periods=3,
        frequency_tolerance_periods=1,
        missing_policy="drop",
        missing_policy_limit=0.4,
        missing_policy_summary="dropped B",
        missing_policy_dropped={"B"},
        missing_policy_filled={"A"},
        date_range=("2020-01-31", "2020-12-31"),
        start="2020-01-31",
        end="2020-12-31",
    )
    return SimpleNamespace(metadata=metadata, frame=frame)


def test_build_validation_report_flags_warnings(schema_module: Any) -> None:
    frame = pd.DataFrame({"A": [1.0] + [None] * 9, "B": [None] * 10})
    validated = _dummy_validated(frame)
    report = schema_module._build_validation_report(validated)

    assert any("quite small" in warning for warning in report["warnings"])
    assert any("missing values" in warning for warning in report["warnings"])
    assert any("Missing-data policy" in warning for warning in report["warnings"])


def test_build_meta_and_validate_df(
    monkeypatch: pytest.MonkeyPatch, schema_module: Any
) -> None:
    frame = pd.DataFrame({"A": [1.0], "B": [2.0]})
    validated = _dummy_validated(frame)
    monkeypatch.setattr(schema_module, "validate_market_data", lambda df: validated)

    result_frame, meta = schema_module._validate_df(frame)
    assert result_frame.equals(frame)
    assert meta["n_rows"] == len(frame)
    assert meta["frequency"] == "Monthly"


def test_load_and_validate_csv(
    monkeypatch: pytest.MonkeyPatch, schema_module: Any
) -> None:
    csv_buffer = io.StringIO("Date,A,B\n2020-01-31,1.0,2.0\n")
    validated = _dummy_validated(pd.DataFrame({"A": [1.0], "B": [2.0]}))
    monkeypatch.setattr(schema_module, "validate_market_data", lambda df: validated)

    frame, meta = schema_module.load_and_validate_csv(csv_buffer)
    assert not frame.empty and meta["symbols"] == ["A", "B"]


def test_load_and_validate_excel(
    monkeypatch: pytest.MonkeyPatch, schema_module: Any
) -> None:
    excel_buffer = io.BytesIO()
    excel_buffer.name = "data.xlsx"  # type: ignore[attr-defined]

    validated = _dummy_validated(pd.DataFrame({"A": [1.0], "B": [2.0]}))

    def fake_read_excel(buf: io.BytesIO) -> pd.DataFrame:
        buf.seek(0)
        return validated.frame

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)
    monkeypatch.setattr(schema_module, "validate_market_data", lambda df: validated)

    frame, meta = schema_module.load_and_validate_file(excel_buffer)
    assert list(frame.columns) == ["A", "B"]
    assert meta["missing_policy"] == "drop"


def test_load_and_validate_unknown_extension(
    monkeypatch: pytest.MonkeyPatch, schema_module: Any
) -> None:
    csv_buffer = io.StringIO("Date,A\n2020-01-31,1.0\n")
    csv_buffer.name = "data.unknown"  # type: ignore[attr-defined]

    validated = _dummy_validated(pd.DataFrame({"A": [1.0]}))
    monkeypatch.setattr(schema_module, "validate_market_data", lambda df: validated)

    frame, _ = schema_module.load_and_validate_file(csv_buffer)
    assert "A" in frame.columns


def test_infer_benchmarks_detects_obvious_candidates(schema_module: Any) -> None:
    candidates = schema_module.infer_benchmarks(["FundA", "SP500 Equity", "agg_bond"])
    assert candidates == ["SP500 Equity", "agg_bond"]
