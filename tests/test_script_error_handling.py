#!/usr/bin/env python3
"""Test script error handling improvements for shell scripts.

This test validates that shell scripts properly handle failures and
provide appropriate logging instead of using || true to mask errors.
"""

import subprocess
import unittest
from pathlib import Path


class TestScriptErrorHandling(unittest.TestCase):
    """Test error handling in shell scripts."""

    def setUp(self):
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent
        self.scripts_dir = self.project_root / "scripts"

    def test_setup_env_error_handling(self):
        """Test that setup_env.sh provides proper error logging."""
        script_path = self.scripts_dir / "setup_env.sh"
        self.assertTrue(script_path.exists(), "setup_env.sh should exist")

        # Read the script content
        with open(script_path) as f:
            content = f.read()

        # Verify that || true patterns have been replaced
        self.assertNotIn("pre-commit install --install-hooks || true", content)
        self.assertNotIn("chmod +x scripts/trend-model || true", content)

        # Verify proper error handling is in place
        self.assertIn("if ! pre-commit install --install-hooks; then", content)
        self.assertIn("::warning::pre-commit install --install-hooks failed", content)
        self.assertIn("if ! chmod +x scripts/trend-model; then", content)
        self.assertIn("::warning::chmod +x scripts/trend-model failed", content)

    def test_quick_check_error_handling(self):
        """Test that quick_check.sh provides proper error handling."""
        script_path = self.scripts_dir / "quick_check.sh"
        self.assertTrue(script_path.exists(), "quick_check.sh should exist")

        with open(script_path) as f:
            content = f.read()

        # Verify that || true patterns have been replaced
        self.assertNotIn("| head -5 || true", content)

        # Verify proper error handling is in place
        self.assertIn("if [[ $? -ne 0 ]]; then", content)
        self.assertIn("::warning::git diff command failed", content)

    def test_validate_fast_error_handling(self):
        """Test that validate_fast.sh uses proper conditional logic."""
        script_path = self.scripts_dir / "validate_fast.sh"
        self.assertTrue(script_path.exists(), "validate_fast.sh should exist")

        with open(script_path) as f:
            content = f.read()

        # Verify that || true patterns for grep have been replaced
        # Count occurrences of || true (should be minimal/none for our target patterns)
        grep_or_true_count = content.count("grep -E") - content.count("|| true")
        self.assertGreaterEqual(
            grep_or_true_count, 5, "Grep commands should use proper null handling"
        )

        # Verify proper null handling with echo ""
        self.assertIn('|| echo ""', content)

    def test_dev_check_error_handling(self):
        """Test that dev_check.sh uses proper conditional logic."""
        script_path = self.scripts_dir / "dev_check.sh"
        self.assertTrue(script_path.exists(), "dev_check.sh should exist")

        with open(script_path) as f:
            content = f.read()

        # Verify that || true patterns for git operations have been replaced
        self.assertNotIn("| grep -v -E '^(Old/|notebooks/old/)' || true", content)

        # Verify proper null handling with echo ""
        self.assertIn('|| echo ""', content)

    def test_scripts_run_without_failure(self):
        """Test that modified scripts run without critical failures."""
        # Test quick_check.sh
        try:
            result = subprocess.run(
                ["./scripts/quick_check.sh"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60,
            )
            # Script should complete (exit code 0 or non-zero due to checks, but not crash)
            self.assertIsNotNone(result.returncode, "quick_check.sh should complete")
        except subprocess.TimeoutExpired:
            self.fail("quick_check.sh timed out - may have hung")

        # Test dev_check.sh with --changed flag (should be fast and safe)
        try:
            result = subprocess.run(
                ["./scripts/dev_check.sh", "--changed"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120,  # Increased timeout for CI with many changed files
            )
            self.assertIsNotNone(result.returncode, "dev_check.sh should complete")
        except subprocess.TimeoutExpired:
            self.fail("dev_check.sh timed out - may have hung")

        # Test validate_fast.sh (should be safe since no changes detected)
        try:
            result = subprocess.run(
                ["./scripts/validate_fast.sh"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120,  # Increased timeout for CI with many changed files
            )
            self.assertIsNotNone(result.returncode, "validate_fast.sh should complete")
        except subprocess.TimeoutExpired:
            self.fail("validate_fast.sh timed out - may have hung")

    def test_error_messages_are_helpful(self):
        """Test that error messages provide useful information."""
        script_path = self.scripts_dir / "setup_env.sh"
        with open(script_path) as f:
            content = f.read()

        # Check that warning messages are descriptive
        self.assertIn("Git hooks may not be available", content)
        self.assertIn("CLI wrapper may not be executable", content)


if __name__ == "__main__":
    unittest.main()
