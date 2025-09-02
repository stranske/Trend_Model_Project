"""Test cases for common human errors in YAML configuration files."""

import sys
import pathlib
import pytest
import tempfile
import os
from hypothesis import given, strategies as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # noqa: E402

from trend_analysis import config  # noqa: E402

# Try to import ValidationError, fallback to ValueError for environments without pydantic
try:
    from pydantic import ValidationError

    ValidationException = ValidationError
except ImportError:
    ValidationException = ValueError


class TestHumanErrors:
    """Test common human errors that occur when editing YAML configuration files."""

    def test_version_as_number(self):
        """Test version field as number (common mistake)."""
        cfg = {"version": 1.0, "data": {}}
        with pytest.raises(
            (ValidationException, ValueError, TypeError),
            match="(version must be a string|str type expected|Input should be a valid string)",
        ):
            config.load_config(cfg)

    def test_version_as_bool(self):
        """Test version field as boolean (YAML parsing issue)."""
        cfg = {"version": True, "data": {}}
        with pytest.raises(
            (ValidationException, ValueError, TypeError),
            match="(version must be a string|str type expected|Input should be a valid string)",
        ):
            config.load_config(cfg)

    def test_empty_version(self):
        """Test empty version string."""
        cfg = {"version": "", "data": {}}
        with pytest.raises(
            (ValidationException, ValueError),
            match="String should have at least 1 character",
        ):
            config.load_config(cfg)

    def test_whitespace_only_version(self):
        """Test whitespace-only version."""
        cfg = {"version": "   ", "data": {}}
        with pytest.raises(
            (ValidationException, ValueError, TypeError),
            match="Version field cannot be empty",
        ):
            config.load_config(cfg)

    def test_data_as_string(self):
        """Test data section as string instead of dict."""
        cfg = {"version": "1.0", "data": "not a dictionary"}
        with pytest.raises(
            (ValidationException, ValueError, TypeError),
            match="data must be a dictionary",
        ):
            config.load_config(cfg)

    def test_preprocessing_as_list(self):
        """Test preprocessing section as list instead of dict."""
        cfg = {"version": "1.0", "data": {}, "preprocessing": ["item1", "item2"]}
        with pytest.raises(
            (ValidationException, ValueError, TypeError),
            match="preprocessing must be a dictionary",
        ):
            config.load_config(cfg)

    def test_vol_adjust_as_number(self):
        """Test vol_adjust section as number instead of dict."""
        cfg = {"version": "1.0", "data": {}, "vol_adjust": 42}
        with pytest.raises(
            (ValidationException, ValueError, TypeError),
            match="vol_adjust must be a dictionary",
        ):
            config.load_config(cfg)

    def test_portfolio_as_string(self):
        """Test portfolio section as string instead of dict."""
        cfg = {"version": "1.0", "data": {}, "portfolio": "invalid"}
        with pytest.raises(
            (ValidationException, ValueError, TypeError),
            match="portfolio must be a dictionary",
        ):
            config.load_config(cfg)

    @pytest.mark.parametrize(
        "section",
        [
            "data",
            "preprocessing",
            "vol_adjust",
            "sample_split",
            "portfolio",
            "metrics",
            "export",
            "run",
        ],
    )
    def test_section_as_null(self, section):
        """Test required sections as null."""
        cfg = {"version": "1.0", "data": {}}
        cfg[section] = None
        with pytest.raises(
            (ValidationException, ValueError, TypeError),
            match=f"{section} must be a dictionary",
        ):
            config.load_config(cfg)

    def test_yaml_file_with_syntax_error(self):
        """Test loading YAML file with syntax errors."""
        yaml_content = """
        version: "1.0"
        data: 
          csv_path: test.csv
        preprocessing: {
          # Missing closing brace - syntax error
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            # YAML parsing will fail with specific error types
            import yaml

            with pytest.raises((yaml.YAMLError, yaml.scanner.ScannerError, Exception)):
                config.load_config(yaml_path)
        finally:
            os.unlink(yaml_path)

    def test_yaml_file_with_validation_error(self):
        """Test loading YAML file with validation errors."""
        yaml_content = """
        version: 1.0  # Should be string
        data: 
          csv_path: test.csv
        preprocessing: "not a dict"  # Should be dict
        vol_adjust: {}
        sample_split: {}
        portfolio: {}
        metrics: {}
        export: {}
        run: {}
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            with pytest.raises((ValidationException, ValueError, TypeError)):
                config.load_config(yaml_path)
        finally:
            os.unlink(yaml_path)

    def test_missing_required_version(self):
        """Test config missing required version field."""
        cfg = {"data": {}}  # Missing version
        with pytest.raises((ValidationException, ValueError, TypeError)):
            config.load_config(cfg)

    @given(st.sampled_from(["  ", "   ", "\t", "\n", " \t ", "\n\n", " \n "]))
    def test_whitespace_version_variants(self, whitespace_version):
        """Test various whitespace-only versions using property-based testing."""
        cfg = {"version": whitespace_version, "data": {}}
        with pytest.raises(
            (ValidationException, ValueError, TypeError),
            match="Version field cannot be empty",
        ):
            config.load_config(cfg)

    @given(st.one_of(st.integers(), st.floats(), st.booleans()))
    def test_version_wrong_types(self, wrong_version):
        """Test version with wrong types using property-based testing."""
        cfg = {"version": wrong_version, "data": {}}
        with pytest.raises(
            (ValidationException, ValueError, TypeError),
            match="(version must be a string|str type expected|Input should be a valid string)",
        ):
            config.load_config(cfg)
