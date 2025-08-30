"""Test for default export constants."""

import pytest

def test_default_constants_exist():
    """Test that default constants are properly defined."""
    from trend_analysis import run_analysis, run_multi_analysis
    
    # Test run_analysis constants
    assert hasattr(run_analysis, 'DEFAULT_OUTPUT_DIRECTORY')
    assert hasattr(run_analysis, 'DEFAULT_OUTPUT_FORMATS')
    assert run_analysis.DEFAULT_OUTPUT_DIRECTORY == "outputs"
    assert run_analysis.DEFAULT_OUTPUT_FORMATS == ["excel"]
    
    # Test run_multi_analysis constants  
    assert hasattr(run_multi_analysis, 'DEFAULT_OUTPUT_DIRECTORY')
    assert hasattr(run_multi_analysis, 'DEFAULT_OUTPUT_FORMATS')
    assert run_multi_analysis.DEFAULT_OUTPUT_DIRECTORY == "outputs"
    assert run_multi_analysis.DEFAULT_OUTPUT_FORMATS == ["excel"]


def test_constants_consistency():
    """Test that constants are consistent between modules."""
    from trend_analysis import run_analysis, run_multi_analysis
    
    assert run_analysis.DEFAULT_OUTPUT_DIRECTORY == run_multi_analysis.DEFAULT_OUTPUT_DIRECTORY
    assert run_analysis.DEFAULT_OUTPUT_FORMATS == run_multi_analysis.DEFAULT_OUTPUT_FORMATS