"""Tests for Streamlit state management."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
import sys
sys.path.append('/home/runner/work/Trend_Model_Project/Trend_Model_Project/app')

import streamlit.state as state_module


class TestSessionState:
    """Test session state management functions."""
    
    def test_initialize_session_state(self):
        """Test session state initialization."""
        mock_state = {}
        with patch('streamlit.state.st.session_state', mock_state):
            state_module.initialize_session_state()
            
            # Check default values are set
            assert "returns_df" in mock_state
            assert "schema_meta" in mock_state
            assert "benchmark_candidates" in mock_state
            assert "validation_report" in mock_state
            assert "upload_status" in mock_state
            assert mock_state["upload_status"] == "pending"
    
    def test_clear_upload_data(self):
        """Test clearing upload data."""
        mock_state = {
            "returns_df": pd.DataFrame({"A": [1, 2]}),
            "schema_meta": {"test": "data"},
            "benchmark_candidates": ["SPX"],
            "validation_report": "test",
            "upload_status": "success"
        }
        
        with patch('streamlit.state.st.session_state', mock_state):
            state_module.clear_upload_data()
            
            # Check data is cleared
            assert "returns_df" not in mock_state
            assert "schema_meta" not in mock_state
            assert "benchmark_candidates" not in mock_state
            assert "validation_report" not in mock_state
            assert mock_state["upload_status"] == "pending"
    
    def test_store_validated_data(self):
        """Test storing validated data."""
        df = pd.DataFrame({"Fund1": [0.01, 0.02]})
        meta = {"test": "metadata"}
        mock_state = {}
        
        with patch('streamlit.state.st.session_state', mock_state):
            state_module.store_validated_data(df, meta)
            
            assert mock_state["returns_df"].equals(df)
            assert mock_state["schema_meta"] == meta
            assert mock_state["upload_status"] == "success"
    
    def test_get_uploaded_data(self):
        """Test retrieving uploaded data."""
        df = pd.DataFrame({"Fund1": [0.01, 0.02]})
        meta = {"test": "metadata"}
        mock_state = {"returns_df": df, "schema_meta": meta}
        
        with patch('streamlit.state.st.session_state', mock_state):
            retrieved_df, retrieved_meta = state_module.get_uploaded_data()
            
            assert retrieved_df.equals(df)
            assert retrieved_meta == meta
    
    def test_has_valid_upload(self):
        """Test checking for valid upload."""
        df = pd.DataFrame({"Fund1": [0.01, 0.02]})
        meta = {"test": "metadata"}
        
        # Test with valid upload
        mock_state = {
            "returns_df": df, 
            "schema_meta": meta, 
            "upload_status": "success"
        }
        with patch('streamlit.state.st.session_state', mock_state):
            assert state_module.has_valid_upload() is True
        
        # Test with missing data
        mock_state = {"upload_status": "success"}
        with patch('streamlit.state.st.session_state', mock_state):
            assert state_module.has_valid_upload() is False
        
        # Test with failed status
        mock_state = {
            "returns_df": df, 
            "schema_meta": meta, 
            "upload_status": "error"
        }
        with patch('streamlit.state.st.session_state', mock_state):
            assert state_module.has_valid_upload() is False
    
    def test_get_upload_summary(self):
        """Test getting upload summary."""
        # Create test data with date index
        dates = pd.date_range("2023-01-01", periods=10, freq="M")
        df = pd.DataFrame({"Fund1": range(10), "Fund2": range(10)}, index=dates)
        meta = {"frequency": "monthly"}
        
        # Test with valid data
        mock_state = {"returns_df": df, "schema_meta": meta}
        with patch('streamlit.state.st.session_state', mock_state):
            summary = state_module.get_upload_summary()
            assert "10 rows × 2 columns" in summary
            assert "Frequency: monthly" in summary
            assert "2023-01-31" in summary  # Start date
        
        # Test with no data
        mock_state = {}
        with patch('streamlit.state.st.session_state', mock_state):
            summary = state_module.get_upload_summary()
            assert summary == "No data uploaded"