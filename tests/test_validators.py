"""Tests for trend_analysis.io.validators module."""

import io

import pandas as pd
import pytest

from trend_analysis.io.validators import (FREQUENCY_MAP, ValidationResult,
                                          create_sample_template,
                                          detect_frequency,
                                          load_and_validate_upload,
                                          validate_returns_schema)


class TestValidationResult:
    """Test ValidationResult class."""

    def test_valid_result_report(self):
        result = ValidationResult(True, [], [], "monthly", ("2023-01-01", "2023-12-31"))
        report = result.get_report()
        assert "✅ Schema validation passed!" in report
        assert "📊 Detected frequency: monthly" in report
        assert "📅 Date range: 2023-01-01 to 2023-12-31" in report

    def test_invalid_result_report(self):
        issues = ["Missing Date column"]
        warnings = ["Too few rows"]
        result = ValidationResult(False, issues, warnings)
        report = result.get_report()
        assert "❌ Schema validation failed!" in report
        assert "Missing Date column" in report
        assert "Too few rows" in report


class TestDetectFrequency:
    """Test frequency detection."""

    def test_monthly_frequency(self):
        dates = pd.date_range("2023-01-31", "2023-12-31", freq="M")
        df = pd.DataFrame(index=dates)
        assert detect_frequency(df) == "monthly"

    def test_daily_frequency(self):
        dates = pd.date_range("2023-01-01", "2023-01-07", freq="D")
        df = pd.DataFrame(index=dates)
        assert detect_frequency(df) == "daily"

    def test_empty_dataframe(self):
        df = pd.DataFrame(index=[])
        assert detect_frequency(df) == "unknown"


class TestValidateReturnsSchema:
    """Test schema validation."""

    def test_valid_schema(self):
        df = pd.DataFrame(
            {
                "Date": ["2023-01-31", "2023-02-28"],
                "Fund1": [0.01, 0.02],
                "Fund2": [0.03, -0.01],
            }
        )
        result = validate_returns_schema(df)
        assert result.is_valid
        assert len(result.issues) == 0
        assert result.frequency is not None

    def test_missing_date_column(self):
        df = pd.DataFrame({"Fund1": [0.01, 0.02], "Fund2": [0.03, -0.01]})
        result = validate_returns_schema(df)
        assert not result.is_valid
        assert "Missing required 'Date' column" in result.issues

    def test_invalid_dates(self):
        df = pd.DataFrame(
            {"Date": ["invalid-date", "2023-02-28"], "Fund1": [0.01, 0.02]}
        )
        result = validate_returns_schema(df)
        assert not result.is_valid
        assert any("invalid dates" in issue for issue in result.issues)

    def test_no_numeric_columns(self):
        df = pd.DataFrame({"Date": ["2023-01-31", "2023-02-28"]})
        result = validate_returns_schema(df)
        assert not result.is_valid
        assert (
            "No numeric return columns found (only Date column present)"
            in result.issues
        )

    def test_duplicate_dates(self):
        df = pd.DataFrame({"Date": ["2023-01-31", "2023-01-31"], "Fund1": [0.01, 0.02]})
        result = validate_returns_schema(df)
        assert not result.is_valid
        assert any("Duplicate dates found" in issue for issue in result.issues)

    def test_warnings_for_missing_values(self):
        # Create a larger dataset to avoid small dataset warning
        dates = ["2023-{:02d}-28".format(i) for i in range(1, 13)]  # 12 months
        fund1_data = [
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]  # 5/12 valid (~42%)
        fund2_data = [0.03 + i * 0.001 for i in range(12)]  # No missing

        df = pd.DataFrame({"Date": dates, "Fund1": fund1_data, "Fund2": fund2_data})
        result = validate_returns_schema(df)
        assert result.is_valid  # Still valid but with warnings
        assert any("has >50% missing values" in warning for warning in result.warnings)


class TestLoadAndValidateUpload:
    """Test file loading and validation."""

    def test_load_csv_file(self):
        csv_data = "Date,Fund1,Fund2\n2023-01-31,0.01,0.02\n2023-02-28,0.03,-0.01"
        file_like = io.StringIO(csv_data)
        file_like.name = "test.csv"

        df, meta = load_and_validate_upload(file_like)
        assert len(df) == 2
        assert len(df.columns) == 2  # Fund1, Fund2
        assert "validation" in meta
        assert meta["validation"].is_valid

    def test_load_invalid_file(self):
        csv_data = "Fund1,Fund2\n0.01,0.02\n0.03,-0.01"  # Missing Date column
        file_like = io.StringIO(csv_data)
        file_like.name = "test.csv"

        with pytest.raises(ValueError) as exc_info:
            load_and_validate_upload(file_like)
        assert "Schema validation failed" in str(exc_info.value)


class TestCreateSampleTemplate:
    """Test sample template creation."""

    def test_create_sample_template(self):
        df = create_sample_template()

        # Check basic structure
        assert "Date" in df.columns
        assert len(df) == 12  # 12 months
        assert len(df.columns) > 1  # Date + return columns

        # Check date format
        dates = pd.to_datetime(df["Date"])
        assert dates.is_monotonic_increasing

        # Check numeric columns
        numeric_cols = [col for col in df.columns if col != "Date"]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col])


class TestFrequencyMap:
    """Test the module-level FREQUENCY_MAP constant."""

    def test_frequency_map_exists(self):
        """Test that FREQUENCY_MAP is defined and contains expected
        mappings."""
        assert isinstance(FREQUENCY_MAP, dict)
        assert len(FREQUENCY_MAP) > 0

    def test_frequency_map_mappings(self):
        """Test that FREQUENCY_MAP contains the expected frequency mappings."""
        expected_mappings = {
            "daily": "D",
            "weekly": "W",
            "monthly": "M",  # For PeriodIndex (pandas >=2.2)
            "quarterly": "Q",
            "annual": "Y",
        }

        for human_readable, pandas_code in expected_mappings.items():
            assert human_readable in FREQUENCY_MAP
            assert FREQUENCY_MAP[human_readable] == pandas_code

    def test_load_and_validate_uses_frequency_map(self):
        """Test that load_and_validate_upload properly uses FREQUENCY_MAP."""
        # Create monthly test data
        dates = pd.date_range("2023-01-31", "2023-12-31", freq="ME")
        df = pd.DataFrame(
            {
                "Date": dates,
                "Fund_A": [
                    0.01,
                    0.02,
                    -0.01,
                    0.015,
                    0.0,
                    0.01,
                    0.008,
                    -0.005,
                    0.02,
                    0.01,
                    -0.01,
                    0.005,
                ],
            }
        )

        # Convert to CSV buffer
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Process the file - should use FREQUENCY_MAP["monthly"] = "M"
        result_df, meta = load_and_validate_upload(csv_buffer)

        # Verify it worked without error and detected monthly frequency
        assert meta["frequency"] == "monthly"
        assert isinstance(result_df.index, pd.DatetimeIndex)
        assert len(result_df) == len(dates)
