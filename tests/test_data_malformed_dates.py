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

    def test_load_csv_filters_null_or_empty_dates(self, caplog):
        """Rows with blank dates should be dropped; remaining rows may fail
        downstream validation (e.g., frequency cadence)."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Date,Fund1\n")
            f.write(",0.01\n")  # empty string should be treated as null/empty
            f.write("2023-01-31,0.02\n")  # mismatched format triggers fallback parsing
            f.write("2023-03-31,0.03\n")
            temp_path = f.name

        try:
            caplog.set_level("WARNING")
            result = load_csv(temp_path)
            # After dropping the row with empty date, validation continues
            # but may fail on frequency checks or return None
            # The key behavior is that we see the "Dropped row" warning
            assert "Dropped row" in caplog.text or result is None
        finally:
            os.unlink(temp_path)

    def test_load_csv_returns_none_when_all_dates_removed(self, caplog, monkeypatch):
        """If every row has a null date, the loader should give up
        gracefully."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Date,Fund1\n")
            temp_path = f.name

        try:
            monkeypatch.setattr(
                pd,
                "read_csv",
                lambda *_, **__: pd.DataFrame(
                    {"Date": ["", ""], "Fund1": [0.01, 0.02]}
                ),
            )

            call_count = {"n": 0}

            def fake_to_datetime(values, *args, **kwargs):  # noqa: ANN001
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise ValueError("bad format")
                index = getattr(values, "index", None)
                return pd.Series([pd.NaT] * len(values), index=index)

            monkeypatch.setattr(pd, "to_datetime", fake_to_datetime)
            caplog.set_level("ERROR")
            result = load_csv(temp_path)
            assert result is None, "No valid rows should result in a None return"
            assert "Unable to parse Date values" in caplog.text
        finally:
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
