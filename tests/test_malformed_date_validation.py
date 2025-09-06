"""Test malformed date validation to ensure they're treated as validation
errors."""

import pandas as pd

from trend_analysis.io.validators import validate_returns_schema


class TestMalformedDateValidation:
    """Test that malformed dates are properly handled as validation errors."""

    def test_malformed_dates_flagged_as_validation_errors(self):
        """Test that malformed dates are detected and flagged as validation
        errors."""
        df = pd.DataFrame(
            {
                "Date": [
                    "2023-01-31",
                    "invalid-date",
                    "2023-03-31",
                    "another-bad-date",
                ],
                "Fund1": [0.01, 0.02, 0.03, 0.04],
                "Fund2": [0.05, 0.06, 0.07, 0.08],
            }
        )
        result = validate_returns_schema(df)
        # Should fail validation due to malformed dates
        assert not result.is_valid
        assert len(result.issues) > 0

        # Check that the error message mentions malformed dates
        error_message = " ".join(result.issues).lower()
        assert "malformed" in error_message
        assert "validation errors" in error_message
        assert "expiration failures" in error_message

        # Should specifically mention the malformed values
        assert "invalid-date" in " ".join(result.issues)
        assert "another-bad-date" in " ".join(result.issues)

    def test_valid_dates_pass_validation(self):
        """Test that valid dates still pass validation."""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-31", "2023-02-28", "2023-03-31"],
                "Fund1": [0.01, 0.02, 0.03],
                "Fund2": [0.05, 0.06, 0.07],
            }
        )

        result = validate_returns_schema(df)

        # Should pass validation with valid dates
        assert result.is_valid
        assert len(result.issues) == 0

    def test_mixed_valid_and_malformed_dates(self):
        """Test behavior with mixed valid and malformed dates."""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-31", "not-a-date", "2023-03-31"],
                "Fund1": [0.01, 0.02, 0.03],
            }
        )

        result = validate_returns_schema(df)

        # Should fail validation due to the one malformed date
        assert not result.is_valid
        assert "1 malformed date(s)" in " ".join(result.issues)
        assert "not-a-date" in " ".join(result.issues)

    def test_all_malformed_dates(self):
        """Test behavior when all dates are malformed."""
        df = pd.DataFrame(
            {
                "Date": ["bad-date-1", "bad-date-2", "bad-date-3"],
                "Fund1": [0.01, 0.02, 0.03],
            }
        )

        result = validate_returns_schema(df)

        # Should fail validation
        assert not result.is_valid
        assert "3 malformed date(s)" in " ".join(result.issues)
        assert "bad-date-1" in " ".join(result.issues)

    def test_empty_date_values_handled(self):
        """Test that empty/null date values are also caught."""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-31", "", "2023-03-31", None],
                "Fund1": [0.01, 0.02, 0.03, 0.04],
            }
        )

        result = validate_returns_schema(df)

        # Should fail validation due to empty/null dates
        assert not result.is_valid
        # Should mention malformed dates (empty strings and None become NaT)
        assert any("malformed" in issue for issue in result.issues)
