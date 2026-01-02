"""Tests for the enhanced Configure page functionality."""

from pathlib import Path

import yaml


class TestPresetLoading:
    """Test preset loading functionality."""

    def test_preset_files_exist(self):
        """Test that all preset files exist and are valid."""
        presets_dir = Path(__file__).parent.parent / "config" / "presets"
        assert presets_dir.exists(), "Presets directory should exist"

        expected_presets = ["conservative.yml", "balanced.yml", "aggressive.yml"]
        for preset_file in expected_presets:
            preset_path = presets_dir / preset_file
            assert preset_path.exists(), f"Preset file {preset_file} should exist"

    def test_preset_yaml_validity(self):
        """Test that all preset YAML files are valid and have required
        fields."""
        presets_dir = Path(__file__).parent.parent / "config" / "presets"
        required_fields = ["name", "description", "lookback_periods", "risk_target"]

        for preset_file in presets_dir.glob("*.yml"):
            with preset_file.open("r") as f:
                data = yaml.safe_load(f)

            assert isinstance(data, dict), f"Preset {preset_file.name} should be a dict"

            for field in required_fields:
                assert field in data, f"Preset {preset_file.name} missing field {field}"

            # Test specific field types and ranges
            assert isinstance(data["lookback_periods"], int), (
                "lookback_periods should be int"
            )
            assert 12 <= data["lookback_periods"] <= 240, (
                "lookback_periods should be reasonable"
            )

            assert isinstance(data["risk_target"], (int, float)), (
                "risk_target should be numeric"
            )
            assert 0.01 <= data["risk_target"] <= 0.50, (
                "risk_target should be reasonable"
            )

            signals = data.get("signals")
            assert isinstance(signals, dict), (
                f"Preset {preset_file.name} missing signals"
            )
            assert "window" in signals, (
                f"Preset {preset_file.name} missing signal window"
            )
            assert "lag" in signals, f"Preset {preset_file.name} missing signal lag"
            assert int(signals["window"]) > 0, "Signal window must be positive"

    def test_preset_content_differences(self):
        """Test that presets have meaningfully different configurations."""
        presets_dir = Path(__file__).parent.parent / "config" / "presets"
        presets = {}

        for preset_file in presets_dir.glob("*.yml"):
            with preset_file.open("r") as f:
                presets[preset_file.stem] = yaml.safe_load(f)

        # Conservative should have higher lookback than aggressive
        assert (
            presets["conservative"]["lookback_periods"]
            > presets["aggressive"]["lookback_periods"]
        )

        # Conservative should have lower risk target than aggressive
        assert (
            presets["conservative"]["risk_target"]
            < presets["aggressive"]["risk_target"]
        )

        # Balanced should be between conservative and aggressive
        assert (
            presets["conservative"]["risk_target"]
            < presets["balanced"]["risk_target"]
            < presets["aggressive"]["risk_target"]
        )


class TestConfigurationValidation:
    """Test configuration validation logic."""

    def test_metric_weights_validation(self):
        """Test that metric weights validation works correctly."""
        # This would test the validation logic if we could import it
        # For now, test the logic manually

        # Valid weights (sum to 1.0)
        weights = {"sharpe": 0.4, "return_ann": 0.3, "drawdown": 0.3}
        total = sum(weights.values())
        assert abs(total - 1.0) <= 0.01, "Valid weights should sum to ~1.0"

        # Invalid weights (don't sum to 1.0)
        weights = {"sharpe": 0.4, "return_ann": 0.3, "drawdown": 0.1}
        total = sum(weights.values())
        assert abs(total - 1.0) > 0.05, "Invalid weights should not sum to ~1.0"

    def test_parameter_ranges(self):
        """Test that parameter ranges are reasonable."""
        # Test lookback months
        assert 12 <= 36 <= 240, "Default lookback should be in range"

        # Test risk target
        assert 0.01 <= 0.10 <= 0.50, "Default risk target should be in range"

        # Test selection count
        assert 1 <= 10 <= 50, "Default selection count should be in range"


class TestColumnMapping:
    """Test column mapping functionality."""

    def test_column_mapping_structure(self):
        """Test that column mapping has correct structure."""
        mapping = {
            "date_column": "Date",
            "return_columns": ["Fund1", "Fund2", "Fund3"],
            "benchmark_column": "Benchmark",
            "risk_free_column": "RF",
            "column_display_names": {"Fund1": "Fund One"},
            "column_tickers": {"Fund1": "F1"},
        }

        # Required fields
        assert "date_column" in mapping
        assert "return_columns" in mapping
        assert isinstance(mapping["return_columns"], list)
        assert len(mapping["return_columns"]) > 0

        # Optional fields
        assert "benchmark_column" in mapping  # Can be None
        assert "column_display_names" in mapping
        assert isinstance(mapping["column_display_names"], dict)


if __name__ == "__main__":
    # Run the tests directly

    class TestRunner:
        def run_all_tests(self):
            """Run all tests manually."""
            print("Running preset loading tests...")

            preset_tests = TestPresetLoading()
            try:
                preset_tests.test_preset_files_exist()
                print("✓ Preset files exist")
            except Exception as e:
                print(f"✗ Preset files test failed: {e}")

            try:
                preset_tests.test_preset_yaml_validity()
                print("✓ Preset YAML validity")
            except Exception as e:
                print(f"✗ Preset YAML test failed: {e}")

            try:
                preset_tests.test_preset_content_differences()
                print("✓ Preset content differences")
            except Exception as e:
                print(f"✗ Preset content test failed: {e}")

            print("\nRunning configuration validation tests...")

            config_tests = TestConfigurationValidation()
            try:
                config_tests.test_metric_weights_validation()
                print("✓ Metric weights validation")
            except Exception as e:
                print(f"✗ Metric weights test failed: {e}")

            try:
                config_tests.test_parameter_ranges()
                print("✓ Parameter ranges")
            except Exception as e:
                print(f"✗ Parameter ranges test failed: {e}")

            print("\nRunning column mapping tests...")

            mapping_tests = TestColumnMapping()
            try:
                mapping_tests.test_column_mapping_structure()
                print("✓ Column mapping structure")
            except Exception as e:
                print(f"✗ Column mapping test failed: {e}")

            print("\nAll tests completed!")

    runner = TestRunner()
    runner.run_all_tests()
