import io
import os
import zipfile
from unittest import mock

import pandas as pd
import pytest

from trend_analysis.io import utils, validators


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


def test_export_bundle_cleans_up_on_zip_failure(tmp_path, monkeypatch):
    """export_bundle should remove incomplete bundles if zipping fails."""
    results = DummyResults()
    zip_path = tmp_path / "bundle_fail.zip"
    before = list(utils._TEMP_FILES_TO_CLEANUP)

    def fake_mkstemp(*_, **__):
        fd = os.open(zip_path, os.O_CREAT | os.O_RDWR)
        return fd, str(zip_path)

    def boom_zipfile(*_, **__):
        raise RuntimeError("zip failure")

    monkeypatch.setattr(utils.tempfile, "mkstemp", fake_mkstemp)
    monkeypatch.setattr(utils.zipfile, "ZipFile", boom_zipfile)

    try:
        with pytest.raises(RuntimeError):
            utils.export_bundle(results, {})
        assert not zip_path.exists()
        assert str(zip_path) not in utils._TEMP_FILES_TO_CLEANUP
    finally:
        utils._TEMP_FILES_TO_CLEANUP[:] = before
        if zip_path.exists():
            os.remove(zip_path)


def test_cleanup_bundle_file_handles_remove_error(tmp_path, monkeypatch):
    """cleanup_bundle_file should swallow removal errors and allow retry."""
    bundle_path = tmp_path / "stubborn.zip"
    bundle_path.write_text("data", encoding="utf-8")
    before = list(utils._TEMP_FILES_TO_CLEANUP)
    utils._TEMP_FILES_TO_CLEANUP.append(str(bundle_path))

    original_remove = os.remove
    call_state = {"count": 0}

    def flaky_remove(path):
        if path == str(bundle_path) and call_state["count"] == 0:
            call_state["count"] += 1
            raise OSError("temp failure")
        return original_remove(path)

    monkeypatch.setattr(utils.os, "remove", flaky_remove)

    try:
        utils.cleanup_bundle_file(str(bundle_path))
        assert str(bundle_path) in utils._TEMP_FILES_TO_CLEANUP
        assert bundle_path.exists()

        utils.cleanup_bundle_file(str(bundle_path))
        assert str(bundle_path) not in utils._TEMP_FILES_TO_CLEANUP
        assert not bundle_path.exists()
    finally:
        utils._TEMP_FILES_TO_CLEANUP[:] = before
        if bundle_path.exists():
            original_remove(str(bundle_path))


def test_cleanup_temp_files_removes_registered_paths(tmp_path):
    """_cleanup_temp_files should delete existing files and clear registry."""
    before = list(utils._TEMP_FILES_TO_CLEANUP)
    file_a = tmp_path / "a.tmp"
    file_b = tmp_path / "b.tmp"
    file_a.write_text("a", encoding="utf-8")
    file_b.write_text("b", encoding="utf-8")
    missing = tmp_path / "missing.tmp"

    try:
        utils._TEMP_FILES_TO_CLEANUP.extend([str(file_a), str(file_b), str(missing)])
        utils._cleanup_temp_files()
        assert not file_a.exists()
        assert not file_b.exists()
        assert utils._TEMP_FILES_TO_CLEANUP == []
    finally:
        utils._TEMP_FILES_TO_CLEANUP[:] = before
        for path in (file_a, file_b):
            if path.exists():
                path.unlink()


def test_cleanup_temp_files_handles_remove_error(tmp_path, monkeypatch):
    """_cleanup_temp_files should ignore removal errors but clear registry."""
    before = list(utils._TEMP_FILES_TO_CLEANUP)
    troublesome = tmp_path / "error.tmp"
    troublesome.write_text("boom", encoding="utf-8")

    original_remove = utils.os.remove

    def flaky_remove(path):
        if path == str(troublesome):
            raise OSError("cannot delete")
        return original_remove(path)

    monkeypatch.setattr(utils.os, "remove", flaky_remove)

    try:
        utils._TEMP_FILES_TO_CLEANUP.append(str(troublesome))
        utils._cleanup_temp_files()
        assert utils._TEMP_FILES_TO_CLEANUP == []
        assert troublesome.exists()
        troublesome.unlink()
    finally:
        utils._TEMP_FILES_TO_CLEANUP[:] = before


def test_export_bundle_zip_failure_missing_file(tmp_path, monkeypatch):
    """export_bundle should cope when cleanup target is already missing."""
    results = DummyResults()
    zip_path = tmp_path / "missing_bundle.zip"
    before = list(utils._TEMP_FILES_TO_CLEANUP)

    def fake_mkstemp(*_, **__):
        fd = os.open(zip_path, os.O_CREAT | os.O_RDWR)
        return fd, str(zip_path)

    def boom_zipfile(*_, **__):
        raise RuntimeError("zip failure")

    original_exists = utils.os.path.exists

    def fake_exists(path):
        if path == str(zip_path):
            return False
        return original_exists(path)

    monkeypatch.setattr(utils.tempfile, "mkstemp", fake_mkstemp)
    monkeypatch.setattr(utils.zipfile, "ZipFile", boom_zipfile)
    monkeypatch.setattr(utils.os.path, "exists", fake_exists)

    try:
        with pytest.raises(RuntimeError):
            utils.export_bundle(results, {})
        # File may remain because cleanup skipped removal
        assert zip_path.exists()
    finally:
        utils._TEMP_FILES_TO_CLEANUP[:] = before
        if zip_path.exists():
            os.remove(zip_path)


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
    assert any("Failed to coerce numeric data" in issue for issue in result.issues)


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
