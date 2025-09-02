"""Test that config module works with and without pydantic."""

import builtins
import sys
from unittest import mock

import pytest


def test_config_import_with_pydantic():
    """Test that config module imports successfully when pydantic is available."""
    from trend_analysis.config.models import Config, _HAS_PYDANTIC

    assert _HAS_PYDANTIC is True

    # Test that Config can be instantiated
    cfg = Config(version="test", data={"key": "value"})
    assert cfg.version == "test"
    assert cfg.data == {"key": "value"}


def test_config_import_without_pydantic():
    """Test that config module imports and works when pydantic is not available."""
    # Mock the import to simulate pydantic not being available
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pydantic" or name.startswith("pydantic."):
            raise ImportError("No module named 'pydantic'")
        return original_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=mock_import):
        # Clear any existing imports
        modules_to_clear = [
            "trend_analysis.config.models",
            "trend_analysis.config",
        ]
        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]

        # Import should work without pydantic
        from trend_analysis.config.models import Config, _HAS_PYDANTIC

        assert _HAS_PYDANTIC is False

        # Test that Config can be instantiated without pydantic
        cfg = Config(version="test", data={"key": "value"})
        assert cfg.version == "test"
        assert cfg.data == {"key": "value"}


def test_config_validation_with_pydantic():
    """Test that validation works when pydantic is available."""
    import importlib
    import trend_analysis.config.models as models

    # Reload to ensure pydantic-backed model is used after previous tests
    models = importlib.reload(models)
    Config = models.Config
    from pydantic import ValidationError

    # Test validation error with pydantic
    with pytest.raises(
        (ValidationError, TypeError),
        match="(Input should be a valid string|version must be a string)",
    ):
        Config(version=123)


def test_config_validation_without_pydantic():
    """Test that validation works when pydantic is not available."""
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pydantic" or name.startswith("pydantic."):
            raise ImportError("No module named 'pydantic'")
        return original_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=mock_import):
        # Clear any existing imports
        modules_to_clear = [
            "trend_analysis.config.models",
            "trend_analysis.config",
        ]
        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]

        from trend_analysis.config.models import Config

        # Test validation error without pydantic
        with pytest.raises((ValueError, TypeError), match="version must be a string"):
            Config(version=123)


def test_load_function_works_without_pydantic():
    """Test that the load function works when pydantic is not available."""
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pydantic" or name.startswith("pydantic."):
            raise ImportError("No module named 'pydantic'")
        return original_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=mock_import):
        # Clear any existing imports
        modules_to_clear = [
            "trend_analysis.config.models",
            "trend_analysis.config",
        ]
        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]

        from trend_analysis.config.models import load

        # Test that loading the default config works
        cfg = load()
        assert hasattr(cfg, "version")
        assert hasattr(cfg, "data")


def test_missing_version_validation_consistency():
    """Test that missing version validation is consistent between modes."""
    # Test with pydantic
    from trend_analysis.config.models import Config

    with pytest.raises(Exception):
        Config(data={})

    # Test without pydantic
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pydantic" or name.startswith("pydantic."):
            raise ImportError("No module named 'pydantic'")
        return original_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=mock_import):
        # Clear any existing imports
        modules_to_clear = [
            "trend_analysis.config.models",
            "trend_analysis.config",
        ]
        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]

        from trend_analysis.config.models import Config

        with pytest.raises(ValueError, match="version field is required"):
            Config(data={})

        # Also test explicit None
        with pytest.raises(ValueError, match="version field is required"):
            Config(version=None, data={})
