"""
Validate that all dependencies required for running tests are present.

This test ensures that the test environment has all necessary tools and
libraries available, including external CLI tools like Node.js and uv.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

# Python packages required for tests
REQUIRED_PYTHON_PACKAGES = [
    "pytest",
    "coverage",
    "hypothesis",
    "pandas",
    "numpy",
    "pydantic",
    "PyYAML",
    "requests",
    "jsonschema",
    "streamlit",
    "fastapi",
    "httpx",
]

# Optional Python packages (tests will skip gracefully if missing)
# Note: pre-commit uses underscore in module name
OPTIONAL_PYTHON_PACKAGES = [
    "black",
    "ruff",
    "mypy",
    "pre_commit",  # Package is 'pre-commit', module is 'pre_commit'
]

# External CLI tools required for full test coverage
REQUIRED_CLI_TOOLS = {
    "node": "Node.js (required for JavaScript workflow tests)",
    "npm": "npm (Node package manager)",
}

# Optional CLI tools (tests will skip if missing)
OPTIONAL_CLI_TOOLS = {
    "uv": "uv (fast Python package installer)",
}


class TestDependencies:
    """Test suite for validating test environment dependencies."""

    def test_python_version(self) -> None:
        """Verify Python version meets minimum requirements."""
        assert sys.version_info >= (
            3,
            11,
        ), f"Python 3.11+ required, found {sys.version_info.major}.{sys.version_info.minor}"

    def test_required_packages_importable(self) -> None:
        """Ensure all required Python packages can be imported."""
        missing = []
        for package in REQUIRED_PYTHON_PACKAGES:
            # Handle package name variations
            import_name = package
            if package == "PyYAML":
                import_name = "yaml"

            try:
                importlib.import_module(import_name)
            except ImportError:
                missing.append(package)

        assert not missing, (
            f"Missing required packages: {', '.join(missing)}\n"
            f"Install with: pip install {' '.join(missing)}"
        )

    def test_optional_packages_documented(self) -> None:
        """Document which optional packages are available."""
        available = []
        missing = []

        for package in OPTIONAL_PYTHON_PACKAGES:
            try:
                importlib.import_module(package)
                available.append(package)
            except ImportError:
                missing.append(package)

        # This test always passes but documents the state
        if missing:
            pytest.skip(
                f"Optional packages not available: {', '.join(missing)}\n"
                f"Some tests may be skipped. Install with: pip install {' '.join(missing)}"
            )

    def test_node_available(self) -> None:
        """Check if Node.js is available for JavaScript tests."""
        node_path = shutil.which("node")
        if not node_path:
            pytest.skip(
                "Node.js not found in PATH. "
                "JavaScript workflow tests will be skipped.\n"
                "Install Node.js: https://nodejs.org/"
            )

        # Get Node.js version
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        version = result.stdout.strip()
        assert version.startswith("v"), f"Unexpected Node.js version format: {version}"

    def test_npm_available_if_node_present(self) -> None:
        """If Node.js is present, npm should also be available."""
        if not shutil.which("node"):
            pytest.skip("Node.js not available, skipping npm check")

        npm_path = shutil.which("npm")
        assert npm_path, (
            "Node.js is installed but npm is not found. "
            "npm is typically bundled with Node.js."
        )

    def test_uv_availability_documented(self) -> None:
        """Document whether uv is available for lockfile tests."""
        uv_path = shutil.which("uv")
        if not uv_path:
            pytest.skip(
                "uv not found in PATH. Lockfile consistency tests will be skipped.\n"
                "Install uv: https://github.com/astral-sh/uv"
            )

    def test_dev_extra_contains_test_tools(self) -> None:
        """Ensure the dev extra declares core testing dependencies."""
        repo_root = Path(__file__).resolve().parents[1]
        pyproject = tomllib.loads(
            (repo_root / "pyproject.toml").read_text(encoding="utf-8")
        )
        operators = ("==", ">=", "<=", "~=", "!=", ">", "<", "===")
        dev_deps = set()

        for entry in pyproject["project"].get("dependencies", []):
            package = entry.split(";")[0].strip()
            for operator in operators:
                if operator in package:
                    package = package.split(operator, 1)[0].strip()
                    break
            dev_deps.add(package.split("[")[0].lower())

        for entry in pyproject["project"]["optional-dependencies"].get("dev", []):
            package = entry.split(";")[0].strip()
            for operator in operators:
                if operator in package:
                    package = package.split(operator, 1)[0].strip()
                    break
            dev_deps.add(package.split("[")[0].lower())

        required = {"pytest", "coverage", "hypothesis"}
        missing = [dep for dep in required if dep not in dev_deps]
        assert not missing, (
            "pyproject.toml missing test tools in [project.optional-dependencies.dev]: "
            + ", ".join(missing)
        )

    def test_pytest_plugins_available(self) -> None:
        """Verify pytest plugins are installed."""
        required_plugins = [
            "pytest_cov",  # pytest-cov
            "pytest_rerunfailures",  # pytest-rerunfailures (if used)
            "hypothesis",  # hypothesis
        ]

        missing = []
        for plugin in required_plugins:
            try:
                importlib.import_module(plugin)
            except ImportError:
                # Check if it's an expected optional plugin
                if plugin not in ["pytest_rerunfailures"]:
                    missing.append(plugin)

        assert not missing, (
            f"Missing required pytest plugins: {', '.join(missing)}\n"
            "Install with: pip install pytest-cov hypothesis"
        )

    def test_coverage_tool_available(self):
        """Verify coverage.py is installed and functional."""
        try:
            import coverage  # noqa: F401
        except ImportError:
            pytest.fail(
                "coverage.py not installed. " "Install with: pip install coverage"
            )

        # Verify coverage CLI is available
        result = subprocess.run(
            ["coverage", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "coverage CLI not functional"

    def test_github_scripts_dependencies(self) -> None:
        """Verify dependencies for GitHub workflow scripts."""
        repo_root = Path(__file__).resolve().parents[1]
        scripts_dir = repo_root / ".github" / "scripts"

        if not scripts_dir.exists():
            pytest.skip(".github/scripts directory not found")

        # Check for JavaScript files that require Node.js
        js_files = list(scripts_dir.glob("*.js"))
        if js_files and not shutil.which("node"):
            pytest.skip(
                f"Found {len(js_files)} JavaScript files in .github/scripts "
                "but Node.js is not available. Install Node.js to run these scripts."
            )

    def test_streamlit_dependencies(self):
        """Verify Streamlit and its dependencies are available."""
        try:
            import streamlit  # noqa: F401
        except ImportError:
            pytest.fail("Streamlit not installed. Required for app tests.")

        # Check for common Streamlit dependencies
        streamlit_deps = ["altair", "pandas", "numpy"]
        missing = []
        for dep in streamlit_deps:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing.append(dep)

        assert not missing, f"Missing Streamlit dependencies: {', '.join(missing)}"




def test_ci_environment_check() -> None:
    """Document the current test environment configuration."""
    import platform

    info = {
        "Python version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "Platform": platform.platform(),
        "Node.js": "available" if shutil.which("node") else "not available",
        "npm": "available" if shutil.which("npm") else "not available",
        "uv": "available" if shutil.which("uv") else "not available",
        "coverage": "available" if shutil.which("coverage") else "not available",
    }

    # This test always passes but logs useful diagnostic info
    print("\n=== Test Environment ===")
    for key, value in info.items():
        print(f"{key}: {value}")
