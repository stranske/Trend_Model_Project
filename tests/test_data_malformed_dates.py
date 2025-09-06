"""Test data loading with malformed dates to ensure proper error handling."""

import os
import tempfile

import pandas as pd
import pytest

from trend_analysis.data import ensure_datetime, load_csv


class TestDataLoadingMalformedDates:
    """Test that data loading properly handles malformed dates."""

    def test_load_csv_with_malformed_dates_returns_none(self):
        """Test that load_csv returns None when malformed dates are
        encountered."""
        # Create a temporary CSV file with malformed dates
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Date,Fund1,Fund2\n")
            f.write("2023-01-31,0.01,0.05\n")
            f.write("invalid-date,0.02,0.06\n")
            f.write("2023-03-31,0.03,0.07\n")
            temp_path = f.name

        try:
            # Should return None due to malformed dates
            result = load_csv(temp_path)
            assert (
                result is None
            ), "load_csv should return None for files with malformed dates"
        finally:
            # Clean up temporary file
            os.unlink(temp_path)

    def test_load_csv_with_valid_dates_succeeds(self):
        """Test that load_csv succeeds with valid dates."""
        # Create a temporary CSV file with valid dates
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Date,Fund1,Fund2\n")
            f.write("01/31/23,0.01,0.05\n")  # mm/dd/yy format
            f.write("02/28/23,0.02,0.06\n")
            f.write("03/31/23,0.03,0.07\n")
            temp_path = f.name

        try:
            # Should succeed with valid dates
            result = load_csv(temp_path)
            assert result is not None, "load_csv should succeed with valid dates"
            assert len(result) == 3
            assert "Date" in result.columns
        finally:
            # Clean up temporary file
            os.unlink(temp_path)

    def test_ensure_datetime_raises_error_for_malformed_dates(self):
        """Test that ensure_datetime raises an error for malformed dates."""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-31", "invalid-date", "2023-03-31"],
                "Fund1": [0.01, 0.02, 0.03],
            }
        )

        # Should raise ValueError due to malformed dates
        with pytest.raises(ValueError) as exc_info:
            ensure_datetime(df, "Date")

        error_message = str(exc_info.value)
        assert "malformed dates" in error_message.lower()
        assert "validation errors" in error_message.lower()
        assert "expiration failures" in error_message.lower()

    def test_ensure_datetime_succeeds_with_valid_dates(self):
        """Test that ensure_datetime succeeds with valid dates."""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-31", "2023-02-28", "2023-03-31"],
                "Fund1": [0.01, 0.02, 0.03],
            }
        )

        # Should succeed and convert to datetime
        result_df = ensure_datetime(df, "Date")
        assert pd.api.types.is_datetime64_any_dtype(result_df["Date"])
        assert len(result_df) == 3

    def test_ensure_datetime_handles_already_datetime_column(self):
        """Test that ensure_datetime handles columns that are already
        datetime."""
        dates = pd.to_datetime(["2023-01-31", "2023-02-28", "2023-03-31"])
        df = pd.DataFrame({"Date": dates, "Fund1": [0.01, 0.02, 0.03]})

        # Should not modify already datetime column
        result_df = ensure_datetime(df, "Date")
        assert pd.api.types.is_datetime64_any_dtype(result_df["Date"])
        assert len(result_df) == 3
        # Dates should be unchanged
        expected_series = pd.Series(dates, name="Date")
        pd.testing.assert_series_equal(result_df["Date"], expected_series)
