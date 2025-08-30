"""Test the io_utils cleanup functionality."""

import sys
import pathlib
import os
import tempfile
import zipfile
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trend_portfolio_app import io_utils  # noqa: E402


def test_export_bundle_creates_temp_files():
    """Test that export_bundle creates temporary files that can be cleaned up."""
    # Create a mock results object with required methods
    mock_results = mock.MagicMock()

    # Patch 'open' so no real files are created
    with mock.patch("builtins.open", mock.mock_open()) as mock_file:
        # Mock the portfolio to return CSV data
        def mock_to_csv(path, header=None):
            with open(path, "w") as f:
                f.write("portfolio_data\n1.0\n2.0\n")

        mock_results.portfolio.to_csv = mock_to_csv

        # Mock event log
        def mock_event_log():
            log_mock = mock.MagicMock()

            def mock_log_to_csv(path):
                with open(path, "w") as f:
                    f.write("event,value\nstart,1\nend,2\n")

            log_mock.to_csv = mock_log_to_csv
            return log_mock

        mock_results.event_log_df = mock_event_log
        mock_results.summary.return_value = {"test": "data"}

        config_dict = {"setting": "value"}

        # Call export_bundle
        zip_path = io_utils.export_bundle(mock_results, config_dict)

        # Verify the ZIP file was created
        assert os.path.exists(zip_path)
        assert zip_path.endswith(".zip")

        # Verify it's a valid ZIP file
        with zipfile.ZipFile(zip_path, "r") as z:
            files_in_zip = z.namelist()
            # Should contain the expected files
            assert "portfolio_returns.csv" in files_in_zip
            assert "event_log.csv" in files_in_zip
            assert "summary.json" in files_in_zip
            assert "config.json" in files_in_zip

        # Cleanup
        io_utils.cleanup_bundle_file(zip_path)

        # Verify file was removed
        assert not os.path.exists(zip_path)


def test_cleanup_bundle_file():
    """Test manual cleanup of bundle files."""
    # Create a temporary file to simulate a bundle
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        temp_path = tmp.name

    # File should exist
    assert os.path.exists(temp_path)

    # Add to cleanup registry
    io_utils._TEMP_FILES_TO_CLEANUP.append(temp_path)

    # Clean it up
    io_utils.cleanup_bundle_file(temp_path)

    # Should be removed from both filesystem and registry
    assert not os.path.exists(temp_path)
    assert temp_path not in io_utils._TEMP_FILES_TO_CLEANUP


def test_cleanup_nonexistent_file():
    """Test that cleanup handles nonexistent files gracefully."""
    # Should not raise an error
    io_utils.cleanup_bundle_file("/nonexistent/file/path.zip")


def test_atexit_cleanup():
    """Test that the atexit cleanup function works."""
    # Create some temporary files
    temp_files = []
    for i in range(3):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_test_{i}.zip") as tmp:
            temp_files.append(tmp.name)

    # Add them to the cleanup registry
    io_utils._TEMP_FILES_TO_CLEANUP.extend(temp_files)

    # Verify they exist
    for f in temp_files:
        assert os.path.exists(f)

    # Call the cleanup function
    io_utils._cleanup_temp_files()

    # Verify they're all gone
    for f in temp_files:
        assert not os.path.exists(f)

    # Registry should be empty
    assert len(io_utils._TEMP_FILES_TO_CLEANUP) == 0
