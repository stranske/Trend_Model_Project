"""Test malformed date validation to ensure they're treated as validation
errors."""

import pandas as pd
import pytest

from trend_analysis.io.market_data import (
    MarketDataValidationError,
    validate_market_data,
)


class TestMalformedDateValidation:
    """Test that malformed dates are properly handled as validation errors."""

    def test_malformed_dates_flagged_as_validation_errors(self, caplog):
        """Test that malformed dates are dropped with warnings."""
        import logging

        df = pd.DataFrame(
            {
                "Date": [
                    "2023-01-31",
                    "invalid-date",
                    "2023-02-28",
                    "another-bad-date",
                ],
                "Fund1": [0.01, 0.02, 0.03, 0.04],
                "Fund2": [0.05, 0.06, 0.07, 0.08],
            }
        )
        with caplog.at_level(logging.WARNING):
            validated = validate_market_data(df)

        assert "Dropped row" in caplog.text
        assert validated.metadata.rows == 2

    def test_valid_dates_pass_validation(self):
        """Test that valid dates still pass validation."""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-31", "2023-02-28", "2023-03-31"],
                "Fund1": [0.01, 0.02, 0.03],
                "Fund2": [0.05, 0.06, 0.07],
            }
        )

        validated = validate_market_data(df)
        assert validated.metadata.rows == 3

    def test_mixed_valid_and_malformed_dates(self, caplog):
        """Test behavior with mixed valid and malformed dates - bad rows are dropped."""
        import logging

        df = pd.DataFrame(
            {
                "Date": ["2023-01-31", "not-a-date", "2023-02-28"],
                "Fund1": [0.01, 0.02, 0.03],
            }
        )

        with caplog.at_level(logging.WARNING):
            validated = validate_market_data(df)

        assert "Dropped row" in caplog.text
        assert validated.metadata.rows == 2

    def test_all_malformed_dates(self, caplog):
        """Test behavior when all dates are malformed - all rows dropped."""
        import logging

        df = pd.DataFrame(
            {
                "Date": ["bad-date-1", "bad-date-2", "bad-date-3"],
                "Fund1": [0.01, 0.02, 0.03],
            }
        )

        with caplog.at_level(logging.WARNING):
            with pytest.raises(MarketDataValidationError):
                validate_market_data(df)

        assert "Dropped row" in caplog.text

    def test_empty_date_values_handled(self, caplog):
        """Test that empty/null date values are dropped with warning."""
        import logging

        df = pd.DataFrame(
            {
                "Date": ["2023-01-31", "2023-02-28", "2023-03-31", None],
                "Fund1": [0.01, 0.02, 0.03, 0.04],
            }
        )

        with caplog.at_level(logging.WARNING):
            validated = validate_market_data(df)

        assert "Dropped row" in caplog.text
        assert validated.metadata.rows == 3
