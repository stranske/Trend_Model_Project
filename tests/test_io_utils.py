import io
import os
import zipfile
from unittest import mock

import pandas as pd
import pytest

from trend_analysis.io import utils
from trend_analysis.io import validators


class DummyResults:
    """Simple results object for export_bundle tests."""

    def __init__(self, portfolio_error=False, event_error=False):
        self.portfolio_error = portfolio_error
        self.event_error = event_error
        self.portfolio = mock.MagicMock()
        self.portfolio.to_csv = self._portfolio_to_csv
        self.event_log_df = self._event_log_df

    def _portfolio_to_csv(self, path, header=None):
        if self.portfolio_error:
            raise RuntimeError("portfolio write failed")
        with open(path, "w") as f:
            f.write("value\n1\n")

    def _event_log_to_csv(self, path):
        if self.event_error:
            raise RuntimeError("event log write failed")
        with open(path, "w") as f:
            f.write("event,value\nstart,1\n")

    def _event_log_df(self):
        log = mock.MagicMock()
        log.to_csv = self._event_log_to_csv
        return log

    def summary(self):
        return {"ok": True}


# ---------------------------------------------------------------------------
# utils.py tests
# ---------------------------------------------------------------------------


def test_export_bundle_with_write_errors(tmp_path):
    """export_bundle should fall back to empty files when writes fail."""
    results = DummyResults(portfolio_error=True, event_error=True)
    zip_path = utils.export_bundle(results, {})

    assert zip_path in utils._TEMP_FILES_TO_CLEANUP
    with zipfile.ZipFile(zip_path) as z:
        # Portfolio write failure -> file with only header
        assert z.read("portfolio_returns.csv").decode("utf-8") == "return\n"
        # Event log failure -> empty file present
        assert z.read("event_log.csv").decode("utf-8") == ""

    utils.cleanup_bundle_file(zip_path)
    assert not os.path.exists(zip_path)
    assert zip_path not in utils._TEMP_FILES_TO_CLEANUP


def test_cleanup_bundle_file_nonexistent(tmp_path):
    """cleanup_bundle_file should ignore missing paths."""
    # Should not raise
    utils.cleanup_bundle_file(tmp_path / "does_not_exist.zip")


# ---------------------------------------------------------------------------
# validators.py tests
# ---------------------------------------------------------------------------


def test_detect_frequency_irregular():
    dates = pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-10"])
    df = pd.DataFrame(index=dates)
    freq = validators.detect_frequency(df)
    assert freq.startswith("irregular")


def test_validate_returns_schema_non_numeric():
    df = pd.DataFrame({"Date": ["2023-01-31", "2023-02-28"], "Fund1": ["a", "b"]})
    result = validators.validate_returns_schema(df)
    assert not result.is_valid
    assert "Column 'Fund1' contains no valid numeric data" in result.issues


def test_load_and_validate_upload_file_not_found(tmp_path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(ValueError) as exc:
        validators.load_and_validate_upload(missing)
    assert "File not found" in str(exc.value)


def test_load_and_validate_upload_parser_error(monkeypatch):
    file_like = io.StringIO("bad,data")
    file_like.name = "bad.csv"

    def boom(*args, **kwargs):
        raise pd.errors.ParserError("bad")

    monkeypatch.setattr(pd, "read_csv", boom)
    with pytest.raises(ValueError) as exc:
        validators.load_and_validate_upload(file_like)
    assert "Failed to parse file" in str(exc.value)
