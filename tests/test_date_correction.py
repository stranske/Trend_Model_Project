"""Tests for date correction helper."""

from __future__ import annotations

import pandas as pd

from streamlit_app.components.date_correction import (
    DateCorrection,
    analyze_date_column,
    apply_date_corrections,
    format_corrections_for_display,
)


class TestDateCorrection:
    """Tests for date correction utilities."""

    def test_november_31_corrected_to_november_30(self):
        """November 31 should be corrected to November 30."""
        df = pd.DataFrame({"Date": ["11/30/2017", "11/31/2017", "12/31/2017"]})
        result = analyze_date_column(df, "Date")

        assert result.has_corrections
        assert len(result.corrections) == 1
        assert result.corrections[0].original_value == "11/31/2017"
        assert result.corrections[0].corrected_value == "11/30/2017"
        assert "November" in result.corrections[0].explanation

    def test_february_30_corrected_to_february_28_or_29(self):
        """February 30 should be corrected to the last day of February."""
        # Non-leap year
        df = pd.DataFrame({"Date": ["02/30/2019"]})
        result = analyze_date_column(df, "Date")

        assert result.has_corrections
        assert result.corrections[0].corrected_value == "02/28/2019"

        # Leap year
        df = pd.DataFrame({"Date": ["02/30/2020"]})
        result = analyze_date_column(df, "Date")

        assert result.has_corrections
        assert result.corrections[0].corrected_value == "02/29/2020"

    def test_september_31_corrected(self):
        """September 31 should be corrected to September 30."""
        df = pd.DataFrame({"Date": ["09/31/2020"]})
        result = analyze_date_column(df, "Date")

        assert result.has_corrections
        assert result.corrections[0].corrected_value == "09/30/2020"

    def test_iso_format_dates_corrected(self):
        """ISO format dates (YYYY-MM-DD) should also be corrected."""
        df = pd.DataFrame({"Date": ["2017-11-31", "2017-12-31"]})
        result = analyze_date_column(df, "Date")

        assert result.has_corrections
        assert len(result.corrections) == 1
        assert result.corrections[0].corrected_value == "2017-11-30"

    def test_completely_invalid_date_not_correctable(self):
        """Completely invalid dates should be marked as unfixable."""
        df = pd.DataFrame({"Date": ["not-a-date", "2020-01-31"]})
        result = analyze_date_column(df, "Date")

        assert not result.has_corrections
        assert result.has_unfixable
        assert len(result.unfixable) == 1
        assert result.unfixable[0][1] == "not-a-date"

    def test_valid_dates_need_no_correction(self):
        """Valid dates should not produce any corrections."""
        df = pd.DataFrame({"Date": ["2020-01-31", "2020-02-29", "2020-03-31"]})
        result = analyze_date_column(df, "Date")

        assert not result.has_corrections
        assert not result.has_unfixable

    def test_mixed_valid_correctable_unfixable(self):
        """Handle mix of valid, correctable, and unfixable dates."""
        df = pd.DataFrame(
            {
                "Date": [
                    "2020-01-31",  # valid
                    "2020-02-30",  # correctable
                    "garbage",  # unfixable
                ]
            }
        )
        result = analyze_date_column(df, "Date")

        assert result.has_corrections
        assert result.has_unfixable
        assert not result.all_fixable
        assert len(result.corrections) == 1
        assert len(result.unfixable) == 1

    def test_apply_corrections(self):
        """Applying corrections should update the DataFrame."""
        df = pd.DataFrame(
            {
                "Date": ["11/30/2017", "11/31/2017", "12/31/2017"],
                "Value": [1, 2, 3],
            }
        )
        result = analyze_date_column(df, "Date")
        corrected_df = apply_date_corrections(df, "Date", result.corrections)

        assert corrected_df.loc[1, "Date"] == "11/30/2017"
        # Original should be unchanged
        assert df.loc[1, "Date"] == "11/31/2017"

    def test_format_corrections_for_display(self):
        """Format corrections should produce readable output."""
        corrections = [
            DateCorrection(
                row_index=87,
                original_value="11/31/2017",
                corrected_value="11/30/2017",
                explanation="November 2017 has 30 days; corrected 31 → 30",
            ),
        ]
        output = format_corrections_for_display(corrections)

        assert "Row 88" in output  # 1-indexed
        assert "11/31/2017" in output
        assert "11/30/2017" in output

    def test_format_corrections_truncates_long_list(self):
        """Long correction lists should be truncated."""
        corrections = [
            DateCorrection(
                row_index=i,
                original_value=f"11/31/201{i % 10}",
                corrected_value=f"11/30/201{i % 10}",
                explanation="test",
            )
            for i in range(15)
        ]
        output = format_corrections_for_display(corrections, max_display=5)

        assert "Row 1" in output
        assert "Row 5" in output
        assert "10 more" in output

    def test_trailing_empty_rows_detected(self):
        """Trailing empty/NaN rows should be detected for removal."""
        df = pd.DataFrame(
            {
                "Date": ["2020-01-31", "2020-02-29", "", "NaN"],
                "Value": [1, 2, 3, 4],
            }
        )
        result = analyze_date_column(df, "Date")

        assert result.has_trailing_empty
        assert len(result.trailing_empty_rows) == 2
        assert 2 in result.trailing_empty_rows
        assert 3 in result.trailing_empty_rows
        assert result.all_fixable  # Trailing empty rows are fixable

    def test_trailing_nan_values_detected(self):
        """Various NaN representations should be detected as trailing."""
        df = pd.DataFrame(
            {
                "Date": ["2020-01-31", "nan", "NaT", "None"],
            }
        )
        result = analyze_date_column(df, "Date")

        assert result.has_trailing_empty
        assert len(result.trailing_empty_rows) == 3

    def test_empty_rows_in_middle_are_droppable(self):
        """Empty rows in the middle of data should be droppable."""
        df = pd.DataFrame(
            {
                "Date": ["2020-01-31", "", "2020-03-31"],
            }
        )
        result = analyze_date_column(df, "Date")

        assert result.has_droppable_empty
        assert not result.has_trailing_empty
        assert not result.has_unfixable
        assert 1 in result.droppable_empty_rows
        assert result.all_fixable  # Can be fixed by dropping

    def test_nan_in_middle_is_droppable(self):
        """NaN values in the middle should be detected as droppable."""
        df = pd.DataFrame(
            {
                "Date": ["2020-01-31", "nan", "2020-03-31", "2020-04-30"],
            }
        )
        result = analyze_date_column(df, "Date")

        assert result.has_droppable_empty
        assert len(result.droppable_empty_rows) == 1
        assert 1 in result.droppable_empty_rows
        assert result.all_fixable

    def test_apply_corrections_drops_trailing_rows(self):
        """Applying corrections should drop trailing empty rows."""
        df = pd.DataFrame(
            {
                "Date": ["2020-01-31", "2020-02-29", "", "NaN"],
                "Value": [1, 2, 3, 4],
            }
        )
        result = analyze_date_column(df, "Date")
        corrected_df = apply_date_corrections(
            df,
            "Date",
            result.corrections,
            drop_rows=result.trailing_empty_rows,
        )

        assert len(corrected_df) == 2
        assert list(corrected_df["Value"]) == [1, 2]

    def test_apply_corrections_drops_middle_empty_rows(self):
        """Applying corrections should drop empty rows from the middle."""
        df = pd.DataFrame(
            {
                "Date": ["2020-01-31", "nan", "2020-03-31", "2020-04-30"],
                "Value": [1, 2, 3, 4],
            }
        )
        result = analyze_date_column(df, "Date")
        all_drops = result.trailing_empty_rows + result.droppable_empty_rows
        corrected_df = apply_date_corrections(
            df,
            "Date",
            result.corrections,
            drop_rows=all_drops,
        )

        assert len(corrected_df) == 3
        assert list(corrected_df["Value"]) == [1, 3, 4]

    def test_format_corrections_shows_trailing_rows(self):
        """Format should show trailing rows that will be removed."""
        output = format_corrections_for_display([], trailing_rows=[10, 11, 12])

        assert "Rows 11–13" in output
        assert "3 trailing empty rows" in output
        assert "will be removed" in output

    def test_day_slightly_over_max_is_correctable(self):
        """Days that are 1-3 over the max should be correctable."""
        # 32nd of any month is 1 day over for 31-day months
        df = pd.DataFrame({"Date": ["01/32/2020"]})
        result = analyze_date_column(df, "Date")

        assert result.has_corrections
        assert result.corrections[0].corrected_value == "01/31/2020"

    def test_day_far_over_max_is_unfixable(self):
        """Days that are way over the max should not be auto-corrected."""
        # 35th is too far off to assume it's a typo
        df = pd.DataFrame({"Date": ["01/35/2020"]})
        result = analyze_date_column(df, "Date")

        assert not result.has_corrections
        assert result.has_unfixable

    def test_handles_european_date_format(self):
        """Should handle DD/MM/YYYY format when day > 12."""
        # 31/11/2017 is clearly DD/MM/YYYY since month can't be 31
        df = pd.DataFrame({"Date": ["31/11/2017"]})
        result = analyze_date_column(df, "Date")

        # This should be detected as November 31 in European format
        # and corrected to 30/11/2017
        assert result.has_corrections
        assert "30" in result.corrections[0].corrected_value
