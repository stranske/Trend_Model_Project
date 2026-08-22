from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trend.input_validation import (
    InputSchema,
    InputValidationError,
    correct_invalid_dates,
    validate_input,
)
from trend_analysis.io.date_correction import analyze_date_column

DATA_DIR = Path(__file__).parent / "data"


def _load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def test_validate_input_accepts_valid_frame() -> None:
    df = _load_csv("valid_input.csv")
    result = validate_input(df)
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tz is not None
    assert len(result) == len(df)


def test_validate_input_missing_required_column() -> None:
    df = _load_csv("missing_column.csv")
    with pytest.raises(InputValidationError, match="Missing required column 'ret'"):
        validate_input(df)


def test_validate_input_drops_unparseable_dates(caplog) -> None:
    """Test that unparseable dates are dropped with warning."""
    import logging

    df = _load_csv("bad_date.csv")
    with caplog.at_level(logging.WARNING):
        result = validate_input(df)
    # Should succeed with the bad row dropped
    assert result is not None
    assert len(result) == 1  # Only the valid row
    assert "Dropped row" in caplog.text


def test_validate_input_autosorts_unsorted_dates() -> None:
    """Unsorted dates are now auto-sorted instead of raising."""
    df = _load_csv("unsorted_dates.csv")
    result = validate_input(df)
    assert result is not None
    # Check dates are sorted ascending - date becomes the index
    if isinstance(result.index, pd.DatetimeIndex):
        assert result.index.is_monotonic_increasing
    elif "date" in result.columns:
        dates = pd.to_datetime(result["date"])
        assert dates.is_monotonic_increasing
    else:
        # If reset_index was called, date may be numeric index
        assert len(result) > 0, "Result should have data"


def test_validate_input_rejects_duplicate_dates() -> None:
    df = _load_csv("duplicate_dates.csv")
    with pytest.raises(InputValidationError, match="Duplicate timestamps"):
        validate_input(df)


def test_validate_input_flags_nan_required_values() -> None:
    df = _load_csv("missing_ret.csv")
    with pytest.raises(InputValidationError, match="Column 'ret' contains missing values"):
        validate_input(df)


def test_validate_input_supports_custom_schema() -> None:
    df = _load_csv("valid_input.csv").rename(columns=str.upper)
    schema = InputSchema(
        date_column="DATE", required_columns=("DATE", "RET"), non_nullable=("RET",)
    )
    result = validate_input(df, schema)
    assert "RET" in result.columns
    assert result.index.name == "DATE"


@pytest.mark.parametrize(
    ("value", "expected", "action"),
    [
        ("11/31/2017", "11/30/2017", "fixed"),
        ("2017-11-31", "2017-11-30", "fixed"),
        ("12/00/2020", "12/01/2020", "fixed"),
        ("12/35/2020", None, "dropped"),
        ("12/31/2201", None, "dropped"),
        ("not-a-date", None, "dropped"),
    ],
)
def test_date_correction_matches_canonical_engine(
    value: str, expected: str | None, action: str
) -> None:
    frame = pd.DataFrame({"Date": ["2020-01-31", value], "ret": [0.1, 0.2]}, index=[10, 20])

    corrected, corrections = correct_invalid_dates(frame, "Date")
    canonical = analyze_date_column(frame.reset_index(drop=True), "Date")

    assert len(corrections) == 1
    assert corrections[0]["action"] == action
    assert corrections[0]["corrected"] == expected
    assert len(corrected) == (2 if action == "fixed" else 1)
    assert corrected.index.tolist() == ([10, 20] if action == "fixed" else [10])
    assert canonical.has_corrections if action == "fixed" else canonical.has_unfixable


def test_date_correction_drops_empty_row_using_canonical_engine() -> None:
    frame = pd.DataFrame({"Date": ["2020-01-31", "", "2020-03-31"], "ret": [0.1, 0.2, 0.3]})

    corrected, corrections = correct_invalid_dates(frame, "Date")

    assert corrected["Date"].tolist() == ["2020-01-31", "2020-03-31"]
    assert corrections == [{"row": 2, "original": "", "corrected": None, "action": "dropped"}]


def test_date_correction_matches_canonical_overflow_tolerance() -> None:
    corrected, corrections = correct_invalid_dates(
        pd.DataFrame({"Date": ["2020-01-31", "01/35/2020"]}), "Date"
    )

    assert len(corrected) == 1
    assert corrections[0]["action"] == "dropped"


def test_correct_invalid_dates_drop_action() -> None:
    frame = pd.DataFrame({"Date": ["2020-01-31", "not-a-date"], "ret": [0.1, 0.2]})

    corrected, corrections = correct_invalid_dates(frame, "Date", action="drop")

    assert corrected["Date"].tolist() == ["2020-01-31"]
    assert corrections == [
        {"row": 2, "original": "not-a-date", "corrected": None, "action": "dropped"}
    ]


def test_correct_invalid_dates_raise_action() -> None:
    frame = pd.DataFrame({"Date": ["2020-01-31", "not-a-date"], "ret": [0.1, 0.2]})

    with pytest.raises(InputValidationError, match="Unable to parse 'Date' at row 2"):
        correct_invalid_dates(frame, "Date", action="raise")
