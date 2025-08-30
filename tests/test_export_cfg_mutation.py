"""Test for export_cfg mutation issue (#476).

This test validates that the load_config function doesn't overwrite existing
export configuration values when processing output configuration.
"""
import tempfile
from pathlib import Path
import yaml

from trend_analysis.config.models import load_config


def test_export_cfg_does_not_overwrite_existing_values():
    """Test that load_config uses setdefault for directory/filename to avoid overwriting existing export values."""
    # Create a config with existing export configuration
    config_data = {
        "version": "1",
        "data": {"csv_path": "test.csv"},
        "preprocessing": {},
        "vol_adjust": {"target_vol": 1.0},
        "sample_split": {"in_start": "2020-01", "in_end": "2020-03", 
                        "out_start": "2020-04", "out_end": "2020-06"},
        "portfolio": {},
        "metrics": {},
        "export": {
            "directory": "/existing/path",  # Existing value that should be preserved
            "filename": "existing_file.xlsx",  # Existing value that should be preserved
        },
        "run": {},
        "output": {
            "path": "/new/path/new_file.xlsx"  # This should NOT overwrite existing directory/filename
        }
    }
    
    # Load config with the problematic output section
    result_config = load_config(config_data)
    
    # Verify that existing export directory/filename values are preserved (not overwritten)
    assert result_config.export["directory"] == "/existing/path"
    assert result_config.export["filename"] == "existing_file.xlsx"


def test_export_cfg_sets_values_when_missing():
    """Test that load_config properly sets export values when they don't exist."""
    config_data = {
        "version": "1",
        "data": {"csv_path": "test.csv"},
        "preprocessing": {},
        "vol_adjust": {"target_vol": 1.0},
        "sample_split": {"in_start": "2020-01", "in_end": "2020-03", 
                        "out_start": "2020-04", "out_end": "2020-06"},
        "portfolio": {},
        "metrics": {},
        "export": {},  # Empty export config - values should be set from output
        "run": {},
        "output": {
            "path": "/new/path/new_file.xlsx"
        }
    }
    
    # Load config 
    result_config = load_config(config_data)
    
    # Verify that export values are set from output when missing, but defaults may apply
    # The key is that existing values in export should not be overwritten
    assert result_config.export["filename"] == "new_file.xlsx"  # This should come from output


def test_export_cfg_partial_overwrite_protection():
    """Test mixed scenario with some existing values and some missing."""
    config_data = {
        "version": "1",
        "data": {"csv_path": "test.csv"},
        "preprocessing": {},
        "vol_adjust": {"target_vol": 1.0},
        "sample_split": {"in_start": "2020-01", "in_end": "2020-03", 
                        "out_start": "2020-04", "out_end": "2020-06"},
        "portfolio": {},
        "metrics": {},
        "export": {
            "directory": "/existing/custom/path",  # This should be preserved
            # filename is missing - should be set from output
            "formats": ["csv", "json"]  # This should be preserved unless output.format overrides
        },
        "run": {},
        "output": {
            "path": "/output/suggested/file.xlsx",
            "format": "excel"  # This SHOULD override existing formats (intended behavior)
        }
    }
    
    # Load config
    result_config = load_config(config_data)
    
    # Verify mixed behavior: 
    # - directory preserved (setdefault behavior)
    # - filename set from output (setdefault behavior)  
    # - formats overridden by output.format (direct assignment behavior)
    assert result_config.export["directory"] == "/existing/custom/path"  # Preserved
    assert result_config.export["filename"] == "file.xlsx"  # Set from output
    assert result_config.export["formats"] == ["excel"]  # Overridden by output.format