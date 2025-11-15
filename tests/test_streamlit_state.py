"""Tests for Streamlit state management."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from streamlit_app import state as state_module


class TestSessionState:
    """Test session state management functions."""

    def test_initialize_session_state(self):
        """Test session state initialization."""
        mock_state = {}
        with patch.object(state_module.st, "session_state", mock_state, create=True):
            state_module.initialize_session_state()

            # Check default values are set
            assert "returns_df" in mock_state
            assert "schema_meta" in mock_state
            assert "benchmark_candidates" in mock_state
            assert "validation_report" in mock_state
            assert "upload_status" in mock_state
            assert mock_state["upload_status"] == "pending"
            assert mock_state["data_hash"] is None
            assert mock_state["data_saved_path"] is None

    def test_clear_upload_data(self):
        """Test clearing upload data."""
        mock_state = {
            "returns_df": pd.DataFrame({"A": [1, 2]}),
            "schema_meta": {"test": "data"},
            "benchmark_candidates": ["SPX"],
            "validation_report": "test",
            "upload_status": "success",
            "data_hash": "old",
            "data_saved_path": "old",
        }

        with patch.object(state_module.st, "session_state", mock_state, create=True):
            state_module.clear_upload_data()

            # Check data is cleared
            assert "returns_df" not in mock_state
            assert "schema_meta" not in mock_state
            assert "benchmark_candidates" not in mock_state
            assert "validation_report" not in mock_state
            assert mock_state["upload_status"] == "pending"
            assert mock_state.get("data_hash") is None
            assert mock_state.get("data_saved_path") is None

    def test_store_validated_data(self):
        """Test storing validated data."""
        df = pd.DataFrame({"Fund1": [0.01, 0.02]})
        meta = {"test": "metadata"}
        mock_state = {}

        with patch.object(state_module.st, "session_state", mock_state, create=True):
            state_module.store_validated_data(
                df, meta, data_hash="hash", saved_path=Path("/tmp/data.csv")
            )

            assert mock_state["returns_df"].equals(df)  # type: ignore[attr-defined]
            assert mock_state["schema_meta"] == meta
            assert mock_state["upload_status"] == "success"
            assert mock_state["data_hash"] == "hash"
            assert mock_state["data_saved_path"] == str(Path("/tmp/data.csv"))

    def test_get_uploaded_data(self):
        """Test retrieving uploaded data."""
        df = pd.DataFrame({"Fund1": [0.01, 0.02]})
        meta = {"test": "metadata"}
        mock_state = {"returns_df": df, "schema_meta": meta}

        with patch.object(state_module.st, "session_state", mock_state, create=True):
            retrieved_df, retrieved_meta = state_module.get_uploaded_data()

            assert retrieved_df.equals(df)  # type: ignore[attr-defined]
            assert retrieved_meta == meta

    def test_has_valid_upload(self):
        """Test checking for valid upload."""
        df = pd.DataFrame({"Fund1": [0.01, 0.02]})
        meta = {"test": "metadata"}

        # Test with valid upload
        mock_state = {"returns_df": df, "schema_meta": meta, "upload_status": "success"}
        with patch.object(state_module.st, "session_state", mock_state, create=True):
            assert state_module.has_valid_upload() is True

        # Test with missing data
        mock_state = {"upload_status": "success"}
        with patch.object(state_module.st, "session_state", mock_state, create=True):
            assert state_module.has_valid_upload() is False

        # Test with failed status
        mock_state = {"returns_df": df, "schema_meta": meta, "upload_status": "error"}
        with patch.object(state_module.st, "session_state", mock_state, create=True):
            assert state_module.has_valid_upload() is False

    def test_get_upload_summary(self):
        """Test getting upload summary."""
        # Create test data with date index
        dates = pd.date_range("2023-01-01", periods=10, freq="ME")
        df = pd.DataFrame({"Fund1": range(10), "Fund2": range(10)}, index=dates)
        meta = {"frequency": "monthly"}

        # Test with valid data
        mock_state = {"returns_df": df, "schema_meta": meta}
        with patch.object(state_module.st, "session_state", mock_state, create=True):
            summary = state_module.get_upload_summary()
            assert "10 rows × 2 columns" in summary
            assert "Frequency: monthly" in summary
            assert "2023-01-31" in summary  # Start date

        # Test with no data
        mock_state = {}
        with patch.object(state_module.st, "session_state", mock_state, create=True):
            summary = state_module.get_upload_summary()
            assert summary == "No data uploaded"
