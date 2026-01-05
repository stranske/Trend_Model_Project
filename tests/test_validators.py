from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from trend_analysis.io.market_data import MarketDataMode, MarketDataValidationError
from trend_analysis.io.validators import (
    ValidationResult,
    create_sample_template,
    detect_frequency,
    load_and_validate_upload,
    validate_returns_schema,
)


class TestValidationResult:
    def test_report_includes_metadata(self) -> None:
        result = ValidationResult(
            True,
            [],
            [],
            frequency="monthly",
            date_range=("2023-01-31", "2023-12-31"),
        )
        result.mode = MarketDataMode.RETURNS
        report = result.get_report()
        assert "✅" in report
        assert "monthly" in report
        assert "2023-12-31" in report
        assert "returns" in report


class TestValidateReturnsSchema:
    def test_valid_dataframe_returns_metadata(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": ["2023-01-31", "2023-02-28", "2023-03-31", "2023-04-30"],
                "FundA": [0.01, 0.02, -0.01, 0.03],
                "FundB": [0.05, 0.01, 0.0, -0.02],
            }
        )
        result = validate_returns_schema(frame)
        assert result.is_valid
        assert result.metadata is not None
        assert result.metadata.mode == MarketDataMode.RETURNS
        assert result.metadata.frequency == "M"
        assert result.metadata.frequency_detected == "M"
        assert result.metadata.frequency_label == "monthly"
        assert result.metadata.frequency_missing_periods == 0
        assert result.metadata.frequency_tolerance_periods >= 0
        assert "small" in result.warnings[0]

    def test_reports_missing_date_column(self) -> None:
        frame = pd.DataFrame({"FundA": [0.01, 0.02]})
        result = validate_returns_schema(frame)
        assert not result.is_valid
        assert any("Missing a 'Date'" in issue for issue in result.issues)

    def test_detects_duplicate_dates(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": ["2023-01-31", "2023-01-31", "2023-02-28"],
                "FundA": [0.01, 0.02, 0.03],
            }
        )
        result = validate_returns_schema(frame)
        assert not result.is_valid
        assert any("duplicate" in issue.lower() for issue in result.issues)

    def test_detects_non_numeric_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": ["2023-01-31", "2023-02-28"],
                "FundA": ["foo", "bar"],
            }
        )
        result = validate_returns_schema(frame)
        assert not result.is_valid
        assert any("no numeric data" in issue for issue in result.issues)

    def test_warns_on_sparse_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2023-01-31", periods=12, freq="ME"),
                "FundA": [0.01] * 4 + [None] * 8,
                "FundB": [0.02] * 12,
            }
        )
        result = validate_returns_schema(frame)
        assert result.is_valid
        assert any("Missing-data policy" in warning for warning in result.warnings)
        assert result.metadata is not None
        assert result.metadata.columns == ["FundB"]

    def test_report_mentions_missing_policy(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2023-01-31", periods=6, freq="ME"),
                "FundA": [0.01, None, None, 0.02, 0.015, 0.01],
                "FundB": [0.02] * 6,
            }
        )
        result = validate_returns_schema(frame)
        report = result.get_report()
        assert "Missing data policy" in report


class TestLoadAndValidateUpload:
    def _make_csv(self, tmp_path: Path) -> Path:
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2023-01-31", periods=6, freq="ME"),
                "FundA": [0.01, 0.02, -0.01, 0.03, 0.01, 0.005],
            }
        )
        csv_path = tmp_path / "data.csv"
        frame.to_csv(csv_path, index=False)
        return csv_path

    def test_reads_csv_buffer(self, tmp_path: Path) -> None:
        csv_path = self._make_csv(tmp_path)
        df, meta = load_and_validate_upload(str(csv_path))
        assert df.attrs["market_data"]["metadata"]["mode"] == "returns"
        metadata = meta["metadata"]
        assert metadata["frequency_label"] == "monthly"
        assert metadata["frequency_detected"] == "M"
        assert meta["frequency"] == "monthly"

    def test_raises_validation_error(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "invalid.csv"
        csv_path.write_text("FundA,FundB\n1,2\n3,4")
        with pytest.raises(MarketDataValidationError):
            load_and_validate_upload(csv_path)

    def test_handles_excel_stream(self, tmp_path: Path) -> None:
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2023-01-31", periods=3, freq="ME"),
                "FundA": [0.01, -0.02, 0.03],
            }
        )
        buf = io.BytesIO()
        frame.to_excel(buf, index=False)
        buf.seek(0)
        buf.name = "upload.xlsx"
        df, meta = load_and_validate_upload(buf)
        assert meta["mode"] == "returns"
        assert len(df) == 3

    def test_file_errors_surface_user_message(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.csv"
        with pytest.raises(ValueError, match="File not found"):
            load_and_validate_upload(missing)

        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="directory"):
                load_and_validate_upload(Path(temp_dir))

    def test_parser_error_is_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_parser(*args, **kwargs):
            raise pd.errors.ParserError("bad")

        monkeypatch.setattr(pd, "read_csv", raise_parser)

        class NamedStringIO(io.StringIO):
            def __init__(self, *args, name=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.name = name

        buffer = NamedStringIO("Date,Fund\n2023-01-31,1.0", name="broken.csv")
        with pytest.raises(ValueError, match="Failed to parse"):
            load_and_validate_upload(buffer)


class TestDetectFrequency:
    def test_daily_frequency(self) -> None:
        index = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(index=index)
        assert detect_frequency(df) == "daily"

    def test_unknown_frequency(self) -> None:
        df = pd.DataFrame(index=[])
        assert detect_frequency(df) == "unknown"


class TestCreateSampleTemplate:
    def test_template_contains_expected_columns(self) -> None:
        template = create_sample_template()
        assert "Date" in template.columns
        assert template.shape[0] == 12
