"""Data validation module for trend analysis uploads."""

from __future__ import annotations

import io
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Map human readable frequency labels to short aliases.  These aliases are
# subsequently normalised to the final pandas ``Period`` codes via
# ``PANDAS_FREQ_MAP``.  Keeping the alias stage allows us to retain backwards
# compatibility (e.g. ``"ME"`` for month-end) while exposing the canonical codes
# through ``FREQUENCY_MAP`` for use throughout the codebase.
FREQ_ALIAS_MAP: Dict[str, str] = {
    "daily": "D",
    "weekly": "W",
    "monthly": "ME",  # mapped to ``M`` via ``PANDAS_FREQ_MAP``
    "quarterly": "Q",
    "annual": "A",  # mapped to ``Y``
}


class ValidationResult:
    """Result of schema validation with detailed feedback."""

    def __init__(
        self,
        is_valid: bool,
        issues: List[str],
        warnings: List[str],
        frequency: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
    ):
        self.is_valid = is_valid
        self.issues = issues
        self.warnings = warnings
        self.frequency = frequency
        self.date_range = date_range

    def get_report(self) -> str:
        """Generate a human-readable validation report."""
        lines = []
        if self.is_valid:
            lines.append("✅ Schema validation passed!")
            if self.frequency:
                lines.append(f"📊 Detected frequency: {self.frequency}")
            if self.date_range:
                lines.append(
                    f"📅 Date range: {self.date_range[0]} to {self.date_range[1]}"
                )
        else:
            lines.append("❌ Schema validation failed!")

        if self.issues:
            lines.append("\n🔴 Issues that must be fixed:")
            for issue in self.issues:
                lines.append(f"  • {issue}")

        if self.warnings:
            lines.append("\n🟡 Warnings:")
            for warning in self.warnings:
                lines.append(f"  • {warning}")

        return "\n".join(lines)


def detect_frequency(df: pd.DataFrame) -> str:
    """Detect the frequency of the time series data."""
    if len(df) < 2:
        return "unknown"

    # Calculate the most common time difference
    diffs = df.index.to_series().diff().dropna()
    if diffs.empty:
        return "unknown"

    # Convert to days for analysis
    days_diffs = diffs.dt.days
    median_diff = days_diffs.median()

    if median_diff <= 1:
        return "daily"
    elif 6 <= median_diff <= 8:
        return "weekly"
    elif 28 <= median_diff <= 35:
        return "monthly"
    elif 88 <= median_diff <= 95:
        return "quarterly"
    elif 360 <= median_diff <= 370:
        return "annual"
    else:
        return f"irregular ({median_diff:.0f} days avg)"


def validate_returns_schema(df: pd.DataFrame) -> ValidationResult:
    """Validate that a DataFrame conforms to the expected returns schema."""
    issues: List[str] = []
    warnings: List[str] = []

    # Check for Date column
    if "Date" not in df.columns:
        issues.append("Missing required 'Date' column")
        return ValidationResult(False, issues, warnings)

    # Try to parse dates
    try:
        date_series = pd.to_datetime(df["Date"])
    except Exception as e:
        issues.append(f"Date column contains invalid dates: {str(e)}")
        return ValidationResult(False, issues, warnings)

    # Check for numeric columns
    non_date_cols = [col for col in df.columns if col != "Date"]
    if not non_date_cols:
        issues.append("No numeric return columns found (only Date column present)")
        return ValidationResult(False, issues, warnings)

    # Validate numeric data
    numeric_issues = []
    for col in non_date_cols:
        try:
            numeric_vals = pd.to_numeric(df[col], errors="coerce")
            non_null_count = numeric_vals.notna().sum()
            if non_null_count == 0:
                numeric_issues.append(f"Column '{col}' contains no valid numeric data")
            elif non_null_count < len(df) * 0.5:
                warnings.append(
                    (
                        f"Column '{col}' has >50% missing values "
                        f"({non_null_count}/{len(df)} valid)"
                    )
                )
        except Exception:
            numeric_issues.append(f"Column '{col}' cannot be converted to numeric")

    if numeric_issues:
        issues.extend(numeric_issues)
        return ValidationResult(False, issues, warnings)

    # Check for duplicates
    if df["Date"].duplicated().any():
        dup_dates = df[df["Date"].duplicated()]["Date"].tolist()
        msg = "Duplicate dates found: " + str(dup_dates[:5])
        if len(dup_dates) > 5:
            msg += "..."
        issues.append(msg)

    # Create a temporary DataFrame for frequency detection
    temp_df = df.copy()
    temp_df["Date"] = date_series
    temp_df = temp_df.set_index("Date").sort_index()

    frequency = detect_frequency(temp_df)
    date_range = (
        temp_df.index.min().strftime("%Y-%m-%d"),
        temp_df.index.max().strftime("%Y-%m-%d"),
    )

    # Additional checks
    if len(temp_df) < 12:
        warnings.append(
            (
                f"Dataset is quite small ({len(temp_df)} periods) - "
                "consider more data for robust analysis"
            )
        )

    is_valid = len(issues) == 0
    return ValidationResult(is_valid, issues, warnings, frequency, date_range)


def load_and_validate_upload(file_like: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load and validate an uploaded file with enhanced validation."""
    # Determine file type
    name = getattr(file_like, "name", "").lower()

    try:
        if name.endswith((".xlsx", ".xls")):
            # Read Excel file
            if hasattr(file_like, "read"):
                data = file_like.read()
                file_like.seek(0)  # Reset for potential re-read
                buf = io.BytesIO(data)
                df = pd.read_excel(buf)
            else:
                df = pd.read_excel(file_like)
        else:
            # Default to CSV
            df = pd.read_csv(file_like)
    except Exception as e:
        raise ValueError(f"Failed to read file: {str(e)}")

    # Validate schema
    validation = validate_returns_schema(df)

    if not validation.is_valid:
        raise ValueError(f"Schema validation failed:\n{validation.get_report()}")

    # Process the data (similar to existing logic)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # Normalize to period-end timestamps using detected frequency
    # Use PeriodIndex.to_timestamp(how='end') to ensure end-of-period alignment
    idx = pd.to_datetime(df.index)
    # Map human-friendly frequency labels (e.g. ``"monthly"``) to pandas
    # ``Period`` codes using ``FREQUENCY_MAP``. Default to monthly if detection
    # failed so downstream code still receives a valid index.
    freq_key = (validation.frequency or "").lower()
    pandas_freq = FREQUENCY_MAP.get(freq_key, "M")
    df.index = pd.PeriodIndex(idx, freq=pandas_freq).to_timestamp(how="end")
    df = df.dropna(axis=1, how="all")

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create metadata
    meta = {
        "original_columns": list(df.columns),
        "n_rows": len(df),
        "validation": validation,
        "frequency": validation.frequency,
        "date_range": validation.date_range,
    }

    return df, meta


def create_sample_template() -> pd.DataFrame:
    """Create a sample returns template DataFrame."""
    # Create a simple monthly returns template
    dates = pd.date_range(start="2023-01-31", end="2023-12-31", freq="ME")

    # Generate some sample return data
    np.random.seed(42)  # For reproducible sample data
    n_funds = 5
    sample_data: Dict[str, Any] = {
        "Date": dates,
    }

    for i in range(1, n_funds + 1):
        # Generate realistic monthly returns (mean ~0.8%, std ~3%)
        returns = np.random.normal(0.008, 0.03, len(dates))
        sample_data[f"Fund_{i:02d}"] = returns

    # Add a benchmark
    benchmark_returns = np.random.normal(0.007, 0.025, len(dates))
    sample_data["SPX_Benchmark"] = benchmark_returns

    return pd.DataFrame(sample_data)
