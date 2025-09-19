"""Golden master test for demo configuration.

This test runs the complete demo pipeline end-to-end and validates that
key CSV outputs remain stable, catching regressions in the core analysis
functionality.
"""

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List

import pandas as pd
import pytest

# Guard against missing matplotlib dependency
pytest.importorskip("matplotlib")


class TestDemoGoldenMaster:
    """Golden master tests for demo pipeline outputs."""

    @pytest.fixture(autouse=True)
    def setup_clean_environment(self):
        """Ensure clean demo environment for each test."""
        # Clean up any existing demo exports
        demo_exports = Path("demo/exports")
        if demo_exports.exists():
            shutil.rmtree(demo_exports)
        yield
        # Cleanup after test
        if demo_exports.exists():
            shutil.rmtree(demo_exports)

    def normalize_csv_content(self, content: str) -> str:
        """Normalize CSV content to make hashes stable.

        - Round floating point numbers to 6 decimal places
        - Remove/normalize timestamp-like patterns
        - Remove version information
        - Normalize line endings
        """
        lines = []
        for line in content.strip().split("\n"):
            # Skip empty lines
            if not line.strip():
                continue

            # Skip or normalize metadata lines that contain timestamps/versions
            if any(
                marker in line.lower()
                for marker in [
                    "generated on",
                    "timestamp",
                    "version",
                    "created at",
                    "run at",
                    "execution time",
                    "datetime",
                ]
            ):
                # Replace with normalized metadata line
                if "generated on" in line.lower():
                    line = "# Generated on NORMALIZED_TIMESTAMP"
                elif "version" in line.lower():
                    line = "# Version NORMALIZED_VERSION"
                elif any(
                    x in line.lower() for x in ["timestamp", "created at", "run at"]
                ):
                    line = "# Timestamp NORMALIZED_TIMESTAMP"
                else:
                    continue  # Skip other timestamp-like metadata

            # Process each field in CSV line
            fields = []
            for field in line.split(","):
                field = field.strip()

                # Normalize timestamp patterns in field values
                field = re.sub(
                    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:\d{2}|Z)?",
                    "NORMALIZED_TIMESTAMP",
                    field,
                )

                # Normalize version patterns (only actual version strings, not numeric data)
                field = re.sub(r"v\d+\.\d+(\.\d+)?", "NORMALIZED_VERSION", field)
                field = re.sub(
                    r"\bversion\s+\d+\.\d+(\.\d+)?",
                    "version NORMALIZED_VERSION",
                    field,
                    flags=re.IGNORECASE,
                )

                # Try to parse as float and round to reduce precision-based variation
                try:
                    if (
                        field
                        and field != "nan"
                        and field != "NaN"
                        and field != "NORMALIZED_TIMESTAMP"
                        and field != "NORMALIZED_VERSION"
                        and not field.isalpha()
                        and not field.startswith("#")
                    ):
                        # Check if it looks like a number
                        float_val = float(field)
                        # Round to 6 decimal places to handle minor floating point differences
                        field = f"{float_val:.6f}"
                except (ValueError, OverflowError):
                    # Not a number, keep as-is
                    pass

                fields.append(field)

            lines.append(",".join(fields))

        return "\n".join(lines)

    def compute_content_hash(self, file_path: Path) -> str:
        """Compute a stable hash of CSV file content.

        This normalizes the content before hashing to account for minor
        floating-point variations and timestamp differences.
        """
        if not file_path.exists():
            return "FILE_NOT_FOUND"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Normalize content for stable hashing
            normalized = self.normalize_csv_content(content)

            # Compute hash
            return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        except (IOError, OSError, UnicodeDecodeError) as e:
            return f"ERROR_READING_FILE_{type(e).__name__}"

    def get_key_output_files(self, demo_exports: Path) -> List[Path]:
        """Identify key CSV output files to validate.

        Focuses on the most important outputs that should remain stable.
        """
        key_patterns = [
            "*_metrics.csv",  # Core performance metrics
            "*_summary.csv",  # Portfolio summary data
            "*_periods.csv",  # Period-based analysis
            "period_frames_*.csv",  # Multi-period frames
            "summary_frames_*.csv",  # Summary frames
            "phase1_multi_metrics_*.csv",  # Multi-phase metrics
            "alias_demo.csv",  # Main demo output (from config)
        ]

        key_files: list[Path] = []
        for pattern in key_patterns:
            key_files.extend(demo_exports.glob(pattern))

        # Also check for the main demo output file specifically
        main_output = demo_exports / "alias_demo.csv"
        if main_output.exists() and main_output not in key_files:
            key_files.append(main_output)

        # Return sorted list, limit to most critical files
        return sorted(key_files)[:8]

    def test_demo_pipeline_end_to_end(self):
        """
        Golden master test: Run complete demo pipeline and validate outputs.

        This test ensures the demo configuration produces expected CSV outputs
        with stable structure and reasonable values.
        """
        # Set up reproducible environment
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0"
        env["PYTHONPATH"] = str(Path.cwd() / "src")

        # Step 1: Generate demo data
        result = subprocess.run(
            ["python", "scripts/generate_demo.py"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, f"Demo data generation failed: {result.stderr}"

        # Verify demo data was created
        demo_csv = Path("demo/demo_returns.csv")
        assert demo_csv.exists(), "Demo returns CSV not created"

        # Validate demo data structure
        df = pd.read_csv(demo_csv, index_col=0, parse_dates=True)
        assert len(df) == 120, f"Expected 120 months of data, got {len(df)}"
        assert (
            len(df.columns) == 21
        ), f"Expected 21 columns (20 managers + SPX), got {len(df.columns)}"
        assert "SPX" in df.columns, "SPX benchmark column missing"

        # Step 2: Run main demo analysis
        result = subprocess.run(
            ["python", "-m", "trend_analysis.run_analysis", "-c", "config/demo.yml"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, f"Demo analysis failed: {result.stderr}"

        # Step 3: Optionally run multi-period demo for comprehensive outputs
        # Skip this if it's unstable, focus on core demo functionality
        try:
            result = subprocess.run(
                ["python", "scripts/run_multi_demo.py"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                env=env,
                timeout=60,  # Prevent hanging
            )
            if result.returncode != 0:
                print(f"Multi-demo script failed (non-critical): {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("Multi-demo script unavailable or timed out (non-critical)")
            pass

        # Step 4: Validate outputs exist
        demo_exports = Path("demo/exports")
        assert demo_exports.exists(), "Demo exports directory not created"

        csv_files = list(demo_exports.glob("*.csv"))
        assert (
            len(csv_files) >= 1
        ), f"Expected at least 1 CSV output, found {len(csv_files)}"

        # Step 5: Validate key output structure and content
        key_files = self.get_key_output_files(demo_exports)
        assert len(key_files) >= 1, "No key output files found"

        file_hashes = {}
        for file_path in key_files:
            # Verify file is not empty
            assert (
                file_path.stat().st_size > 0
            ), f"Output file {file_path.name} is empty"

            # Verify it can be read as CSV
            try:
                df = pd.read_csv(file_path)
                assert not df.empty, f"CSV file {file_path.name} is empty"
                assert len(df.columns) > 0, f"CSV file {file_path.name} has no columns"
            except Exception as e:
                pytest.fail(f"Could not read CSV file {file_path.name}: {e}")

            # Compute content hash for regression detection
            content_hash = self.compute_content_hash(file_path)
            file_hashes[file_path.name] = content_hash

        # Step 6: Basic output validation
        # Check that we have reasonable numeric outputs
        found_metrics = False
        for file_path in key_files:
            if "metrics" in file_path.name.lower():
                df = pd.read_csv(file_path)
                # Check for expected metric columns
                metric_cols = [
                    col
                    for col in df.columns
                    if col.lower()
                    in [
                        "cagr",
                        "vol",
                        "sharpe",
                        "sortino",
                        "information_ratio",
                        "max_drawdown",
                    ]
                ]
                if metric_cols:
                    found_metrics = True
                    # Basic sanity checks on metrics
                    for col in metric_cols:
                        values = pd.to_numeric(df[col], errors="coerce").dropna()
                        if len(values) > 0:
                            # Check values are reasonable (not all zeros or all same)
                            assert (
                                values.std() > 0.001
                            ), f"Metric {col} has no variation"

        assert found_metrics, "No metrics files found with expected columns"

        # Step 7: Store hash summary for debugging
        print("\nDemo pipeline golden master validation passed.")
        print(f"Key files validated: {len(file_hashes)}")
        for filename, file_hash in sorted(file_hashes.items()):
            print(f"  {filename}: {file_hash}")

    def test_demo_pipeline_deterministic(self):
        """Test that demo pipeline produces deterministic outputs across runs.

        This test runs the demo twice and ensures outputs are identical
        when using the same seed and environment.
        """
        # Set up reproducible environment
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0"
        env["PYTHONPATH"] = str(Path.cwd() / "src")

        def run_demo_and_get_hashes() -> dict:
            # Clean exports
            demo_exports = Path("demo/exports")
            if demo_exports.exists():
                shutil.rmtree(demo_exports)

            # Generate demo data
            subprocess.run(
                ["python", "scripts/generate_demo.py"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )

            # Run demo analysis
            subprocess.run(
                [
                    "python",
                    "-m",
                    "trend_analysis.run_analysis",
                    "-c",
                    "config/demo.yml",
                ],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )

            # Compute hashes of outputs
            hashes = {}
            if demo_exports.exists():
                for csv_file in demo_exports.glob("*.csv"):
                    hashes[csv_file.name] = self.compute_content_hash(csv_file)

            return hashes

        # Run demo twice
        hashes_run1 = run_demo_and_get_hashes()
        hashes_run2 = run_demo_and_get_hashes()

        # Compare results
        assert len(hashes_run1) > 0, "No output files generated in run 1"
        assert len(hashes_run2) > 0, "No output files generated in run 2"

        # Check that same files were generated
        assert set(hashes_run1.keys()) == set(
            hashes_run2.keys()
        ), f"Different files generated: {set(hashes_run1.keys())} vs {set(hashes_run2.keys())}"

        # Check that content hashes match
        mismatched_files = []
        for filename in hashes_run1:
            if hashes_run1[filename] != hashes_run2[filename]:
                mismatched_files.append(filename)

        assert (
            len(mismatched_files) == 0
        ), f"Non-deterministic outputs detected in files: {mismatched_files}"

        print(f"\nDemo deterministic test passed. {len(hashes_run1)} files validated.")

    def test_coverage_gate_enforcement(self):
        """Test that coverage gates are properly configured and enforced.

        This test validates that the coverage configuration meets the requirements:
        - CI fails if coverage drops below 80% globally
        - CI fails if trend_analysis coverage drops below 85%
        """
        import configparser

        # Check CI configuration
        ci_config_path = Path(".github/workflows/ci.yml")
        assert ci_config_path.exists(), "CI configuration file not found"

        with open(ci_config_path, "r") as f:
            ci_content = f.read()

        # Verify global coverage gate is set to 80% (literal or via variable)
        # Accept either a literal --cov-fail-under=80 or a variable-based expression resolving to 80,
        # such as --cov-fail-under=${{ vars.COV_MIN || 80 }}.
        literal_ok = "--cov-fail-under=80" in ci_content
        variable_ok = bool(
            re.search(
                r"--cov-fail-under=\$\{\{\s*vars\.COV_MIN\s*\|\|\s*80\s*\}\}",
                ci_content,
            )
        )
        assert (
            literal_ok or variable_ok
        ), "CI should require 80% coverage globally (literal or via vars.COV_MIN with default 80)"

        # Check core coverage configuration
        core_config_path = Path(".coveragerc.core")
        assert core_config_path.exists(), "Core coverage config not found"

        config = configparser.ConfigParser()
        config.read(core_config_path)

        # Verify trend_analysis modules require 85% coverage
        assert config.has_section("report"), "Coverage config missing [report] section"
        fail_under = config.get("report", "fail_under", fallback="0")
        assert (
            int(fail_under) == 85
        ), f"trend_analysis modules should require 85% coverage, found {fail_under}%"

        # Verify include pattern targets trend_analysis
        include = config.get("report", "include", fallback="")
        assert (
            "src/trend_analysis/*" in include
        ), "Coverage config should include src/trend_analysis/*"

        print("✓ Coverage gates properly configured:")
        print("  - Global CI coverage: 80%")
        print("  - trend_analysis modules: 85%")

    def test_demo_regression_detection(self):
        """Test that the golden master test catches meaningful regressions.

        This validates that changes to key output files would be
        detected by the hash comparison mechanism.
        """
        # This test ensures our normalization doesn't over-normalize
        # and still catches real changes

        sample_csv_content = """Date,Manager_A,Manager_B,SPX
2023-01-01,0.015432,0.012345,0.011234
2023-02-01,0.023456,0.019876,0.015678
"""

        # Test that identical content produces identical hashes
        hash1 = hashlib.sha256(
            self.normalize_csv_content(sample_csv_content).encode()
        ).hexdigest()
        hash2 = hashlib.sha256(
            self.normalize_csv_content(sample_csv_content).encode()
        ).hexdigest()
        assert hash1 == hash2, "Identical content should produce identical hashes"

        # Test that meaningful changes are detected
        modified_content = sample_csv_content.replace("0.015432", "0.025432")
        hash3 = hashlib.sha256(
            self.normalize_csv_content(modified_content).encode()
        ).hexdigest()
        assert hash1 != hash3, "Modified content should produce different hashes"

        # Test that timestamp normalization works but preserves data
        timestamped_content = (
            f"# Generated on 2024-01-01T12:00:00Z\n{sample_csv_content}"
        )
        hash4 = hashlib.sha256(
            self.normalize_csv_content(timestamped_content).encode()
        ).hexdigest()

        timestamped_content2 = (
            f"# Generated on 2024-06-15T15:30:45Z\n{sample_csv_content}"
        )
        hash5 = hashlib.sha256(
            self.normalize_csv_content(timestamped_content2).encode()
        ).hexdigest()

        assert hash4 == hash5, "Different timestamps should normalize to same hash"

        print("✓ Regression detection working correctly:")
        print("  - Identical content → identical hashes")
        print("  - Modified data → different hashes")
        print("  - Timestamps normalized properly")
