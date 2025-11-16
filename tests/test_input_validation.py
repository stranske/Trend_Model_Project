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


def test_validate_input_rejects_unparseable_dates() -> None:
    df = _load_csv("bad_date.csv")
    with pytest.raises(InputValidationError, match="Unable to parse 'date' at row 2"):
        validate_input(df)


def test_validate_input_detects_unsorted_dates() -> None:
    df = _load_csv("unsorted_dates.csv")
    with pytest.raises(InputValidationError, match="sorted in ascending order"):
        validate_input(df)


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
