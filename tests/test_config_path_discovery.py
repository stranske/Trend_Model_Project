"""Tests for robust config path discovery functionality."""

import tempfile
from pathlib import Path
from unittest import mock

import pytest

from trend_analysis.config import DEFAULTS
from trend_analysis.config.models import _find_config_directory
from trend_analysis.gui.utils import list_builtin_cfgs


def test_config_directory_discovery():
    """Test that config directory is found correctly."""
    # Should find the actual config directory
    config_dir = _find_config_directory()
    assert config_dir.is_dir()
    assert (config_dir / "defaults.yml").exists()


def test_defaults_path_exists():
    """Test that DEFAULTS points to an existing file."""
    assert DEFAULTS.exists()
    assert DEFAULTS.name == "defaults.yml"


def test_config_discovery_robustness():
    """Test config discovery works from different locations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a mock directory structure
        mock_project = temp_path / "mock_project"
        mock_config = mock_project / "config"
        mock_src = mock_project / "src" / "trend_analysis" / "config"

        mock_src.mkdir(parents=True)
        mock_config.mkdir(parents=True)
        (mock_config / "defaults.yml").write_text("version: test")

        # Create a mock models.py file in the config package
        mock_models = mock_src / "models.py"

        # Mock the __file__ variable to point to our mock location
        with mock.patch("trend_analysis.config.models.__file__", str(mock_models)):
            # Import and test the function from the mocked location
            from trend_analysis.config.models import (
                _find_config_directory as mock_find_config,
            )

            found_config = mock_find_config()
            assert found_config == mock_config
            assert (found_config / "defaults.yml").exists()


def test_gui_utils_config_discovery():
    """Test that GUI utils can find configs."""
    configs = list_builtin_cfgs()
    assert isinstance(configs, list)
    assert "defaults" in configs  # Should always have defaults.yml
    assert len(configs) > 0


def test_config_discovery_fallback():
    """Test fallback behavior when config directory not found in normal
    search."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a structure where only the fallback path works
        mock_project = temp_path / "mock_project"
        fallback_config = mock_project / "config"
        deep_path = mock_project / "src" / "trend_analysis" / "config"

        deep_path.mkdir(parents=True)
        fallback_config.mkdir(parents=True)
        (fallback_config / "defaults.yml").write_text("version: test")

        mock_models = deep_path / "models.py"

        with mock.patch("trend_analysis.config.models.__file__", str(mock_models)):
            from trend_analysis.config.models import (
                _find_config_directory as mock_find_config,
            )

            # Should find the fallback config
            found_config = mock_find_config()
            assert found_config == fallback_config


def test_config_discovery_failure():
    """Test that appropriate error is raised when no config found."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a structure with no config directory
        deep_path = temp_path / "some" / "deep" / "path"
        deep_path.mkdir(parents=True)
        mock_models = deep_path / "models.py"

        with mock.patch("trend_analysis.config.models.__file__", str(mock_models)):
            from trend_analysis.config.models import (
                _find_config_directory as mock_find_config,
            )

            with pytest.raises(FileNotFoundError, match="Could not find 'config' directory"):
                mock_find_config()
