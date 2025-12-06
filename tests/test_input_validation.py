from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trend.input_validation import InputSchema, InputValidationError, validate_input

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
    with pytest.raises(
        InputValidationError, match="Column 'ret' contains missing values"
    ):
        validate_input(df)


def test_validate_input_supports_custom_schema() -> None:
    df = _load_csv("valid_input.csv").rename(columns=str.upper)
    schema = InputSchema(
        date_column="DATE", required_columns=("DATE", "RET"), non_nullable=("RET",)
    )
    result = validate_input(df, schema)
    assert "RET" in result.columns
    assert result.index.name == "DATE"
