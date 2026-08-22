from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from trend_analysis.io.market_data import (
    MarketDataMode,
    MarketDataValidationError,
    classify_frequency,
    validate_market_data,
)
from trend_analysis.io.validators import (
    create_sample_template,
    load_and_validate_upload,
)


class TestValidateMarketData:
    def test_valid_dataframe_returns_metadata(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": ["2023-01-31", "2023-02-28", "2023-03-31", "2023-04-30"],
                "FundA": [0.01, 0.02, -0.01, 0.03],
                "FundB": [0.05, 0.01, 0.0, -0.02],
            }
        )
        validated = validate_market_data(frame)
        assert validated.metadata.mode == MarketDataMode.RETURNS
        assert validated.metadata.frequency == "M"
        assert validated.metadata.frequency_detected == "M"
        assert validated.metadata.frequency_label == "monthly"
        assert validated.metadata.frequency_missing_periods == 0
        assert validated.metadata.frequency_tolerance_periods >= 0

    def test_reports_missing_date_column(self) -> None:
        frame = pd.DataFrame({"FundA": [0.01, 0.02]})
        with pytest.raises(MarketDataValidationError) as excinfo:
            validate_market_data(frame)
        assert any("Missing a 'Date'" in issue for issue in excinfo.value.issues)

    def test_detects_duplicate_dates(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": ["2023-01-31", "2023-01-31", "2023-02-28"],
                "FundA": [0.01, 0.02, 0.03],
            }
        )
        with pytest.raises(MarketDataValidationError) as excinfo:
            validate_market_data(frame)
        assert any("duplicate" in issue.lower() for issue in excinfo.value.issues)

    def test_detects_non_numeric_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": ["2023-01-31", "2023-02-28"],
                "FundA": ["foo", "bar"],
            }
        )
        with pytest.raises(MarketDataValidationError) as excinfo:
            validate_market_data(frame)
        assert any("no numeric data" in issue for issue in excinfo.value.issues)

    def test_warns_on_sparse_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2023-01-31", periods=12, freq="ME"),
                "FundA": [0.01] * 4 + [None] * 8,
                "FundB": [0.02] * 12,
            }
        )
        validated = validate_market_data(frame)
        assert validated.metadata.columns == ["FundB"]


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

    def test_load_csv_upload(self, tmp_path: Path) -> None:
        csv_path = self._make_csv(tmp_path)
        df, meta = load_and_validate_upload(str(csv_path))
        assert len(df) == 6
        assert meta["n_rows"] == 6
        assert meta["validation"]["is_valid"] is True

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "missing.csv"
        with pytest.raises(ValueError, match="File not found"):
            load_and_validate_upload(csv_path)

    def test_buffer_upload(self, tmp_path: Path) -> None:
        csv_path = self._make_csv(tmp_path)
        buf = io.StringIO(csv_path.read_text())
        buf.name = "data.csv"
        df, meta = load_and_validate_upload(buf)
        assert len(df) == 6
        assert meta["validation"]["issues"] == []

    def test_directory_path_raises(self, tmp_path: Path) -> None:
        with tempfile.TemporaryDirectory(dir=tmp_path) as temp_dir:
            with pytest.raises(ValueError, match="directory"):
                load_and_validate_upload(Path(temp_dir))

    def test_parser_error_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class NamedStringIO(io.StringIO):
            def __init__(self, *args, name=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.name = name

        buffer = NamedStringIO("Date,Fund\n2023-01-31,1.0", name="broken.csv")
        monkeypatch.setattr(
            pd,
            "read_csv",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(pd.errors.ParserError("bad csv")),
        )
        with pytest.raises(ValueError, match="Failed to parse"):
            load_and_validate_upload(buffer)


class TestClassifyFrequency:
    def test_daily_frequency(self) -> None:
        index = pd.date_range("2023-01-01", periods=5, freq="D")
        info = classify_frequency(index)
        assert info["label"] == "daily"

    def test_unknown_frequency(self) -> None:
        info = classify_frequency(pd.DatetimeIndex([]))
        assert info["label"] == "unknown"


class TestCreateSampleTemplate:
    def test_template_contains_expected_columns(self) -> None:
        template = create_sample_template()
        assert "Date" in template.columns
        assert template.shape[0] == 12
