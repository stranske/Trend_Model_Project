"""Tests for robust path resolution functionality."""

import os
import tempfile
from pathlib import Path
import pytest

from trend_analysis.config import find_project_root, DEFAULTS


def test_find_project_root_with_pyproject_toml():
    """Test finding project root using pyproject.toml marker."""
    # Should find the real project root
    root = find_project_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "requirements.txt").exists()
    assert (root / "config").is_dir()
    assert (root / "src").is_dir()


def test_find_project_root_with_custom_start_path():
    """Test finding project root from a nested path."""
    # Start from a nested path and find root
    nested_path = Path(__file__).parent / "nested" / "deeply" / "nested"
    root = find_project_root(start_path=Path(__file__).parent)
    assert (root / "pyproject.toml").exists()


def test_find_project_root_with_environment_override(monkeypatch, tmp_path):
    """Test environment variable override for project root."""
    # Create a fake project structure
    fake_root = tmp_path / "fake_project"
    fake_root.mkdir()
    (fake_root / "config").mkdir()
    (fake_root / "src").mkdir()
    
    # Set environment variable
    monkeypatch.setenv("TREND_PROJECT_ROOT", str(fake_root))
    
    # Should return the environment variable path
    root = find_project_root()
    assert root == fake_root


def test_find_project_root_fallback_structure(tmp_path):
    """Test fallback when no markers found but config/src dirs exist."""
    # Create a structure without markers but with config/src
    fake_root = tmp_path / "no_markers"
    fake_root.mkdir()
    (fake_root / "config").mkdir()
    (fake_root / "src").mkdir()
    nested = fake_root / "src" / "nested"
    nested.mkdir(parents=True)
    
    # Should find root via fallback logic
    root = find_project_root(start_path=nested)
    assert root == fake_root


def test_find_project_root_not_found():
    """Test error when project root cannot be found."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        isolated_path = Path(tmp_dir) / "isolated"
        isolated_path.mkdir()
        
        with pytest.raises(FileNotFoundError, match="Cannot find project root"):
            find_project_root(start_path=isolated_path)


def test_defaults_path_exists():
    """Test that DEFAULTS path is resolved correctly and exists."""
    assert DEFAULTS.exists()
    assert DEFAULTS.name == "defaults.yml"
    assert DEFAULTS.parent.name == "config"


def test_defaults_path_is_absolute():
    """Test that DEFAULTS path is absolute."""
    assert DEFAULTS.is_absolute()


def test_robust_path_vs_hardcoded():
    """Test that robust path resolution gives same result as working hardcoded path."""
    # Get the current working directory (should be project root)
    current_root = find_project_root()
    
    # This should be the same as what the old hardcoded path would resolve to
    # from the config.py file location
    config_file_path = current_root / "src" / "trend_analysis" / "config.py"
    hardcoded_root = config_file_path.resolve().parents[2]
    
    assert current_root == hardcoded_root


def test_environment_variable_override_config_path(monkeypatch, tmp_path):
    """Test that TREND_CFG environment variable still works with robust paths."""
    from trend_analysis.config import load
    
    # Create a custom config file
    custom_config = tmp_path / "custom.yml"
    custom_config.write_text("""
version: "test-version"
data: {}
preprocessing: {}
vol_adjust: {}
sample_split: {}
portfolio: {}
metrics: {}
export: {}
run: {}
""")
    
    # Set environment variable
    monkeypatch.setenv("TREND_CFG", str(custom_config))
    
    # Load config should use the environment variable
    cfg = load()
    assert cfg.version == "test-version"
    
    # Clean up
    monkeypatch.delenv("TREND_CFG", raising=False)


def test_import_find_project_root_from_multiple_modules():
    """Test that find_project_root can be imported from different modules."""
    # Should be available from config module
    from trend_analysis.config import find_project_root as config_find_root
    
    # Should be available from gui.utils 
    from trend_analysis.gui.utils import find_project_root as gui_find_root
    
    # Should be available from conftest (if run as module)
    import tests.conftest
    conftest_find_root = tests.conftest.find_project_root
    
    # All should find the same root
    root1 = config_find_root()
    root2 = gui_find_root()  
    root3 = conftest_find_root()
    
    assert root1 == root2 == root3


if __name__ == "__main__":
    pytest.main([__file__])