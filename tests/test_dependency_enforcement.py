"""
Enforce that all test dependencies are declared and installable.

This test ensures that:
1. All Python imports in test files are declared in pyproject.toml
2. All external CLI tools used in tests are documented
3. The CI environment has all necessary dependencies installed
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path
from typing import Set

import pytest

# Stdlib modules that don't need to be installed
STDLIB_MODULES = {
    "abc",
    "argparse",
    "ast",
    "asyncio",
    "base64",
    "builtins",
    "collections",
    "contextlib",
    "configparser",
    "copy",
    "csv",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "gc",
    "glob",
    "hashlib",
    "importlib",
    "inspect",
    "io",
    "itertools",
    "json",
    "logging",
    "math",
    "multiprocessing",
    "os",
    "pathlib",
    "pkgutil",
    "pickle",
    "platform",
    "random",
    "re",
    "runpy",
    "shlex",
    "shutil",
    "signal",
    "sitecustomize",
    "socket",
    "stat",
    "string",
    "struct",
    "subprocess",
    "sys",
    "tempfile",
    "textwrap",
    "threading",
    "time",
    "tomllib",
    "typing",
    "unittest",
    "urllib",
    "uuid",
    "warnings",
    "weakref",
    "xml",
    "zipfile",
    "__future__",
    "dataclasses",
    "enum",
    "types",
    "traceback",
    "pprint",
}

# Known test framework modules
TEST_FRAMEWORK_MODULES = {
    "pytest",
    "hypothesis",
    "_pytest",
    "pluggy",
}

# Project modules (installed via pip install -e .)
PROJECT_MODULES = {
    "trend_analysis",
    "trend_portfolio_app",
    "streamlit_app",
    "trend_model",
    "trend",
    "data",
    "app",
    "tools",
    "scripts",
    "tests",
    # Test-specific internal modules
    "_autofix_diag",
    "gate_summary",
    "restore_branch_snapshots",
    "test_test_dependencies",
    "decode_raw_input",
    "fallback_split",
    "parse_chatgpt_topics",
    "health_summarize",  # .github/scripts/health_summarize.py
}


_SPECIFIER_PATTERN = re.compile(r"[!=<>~]")


def _extract_requirement_name(entry: str) -> str | None:
    """Return the canonical package name for a pyproject requirement."""

    cleaned = entry.split(";")[0].strip().strip(",")
    if not cleaned:
        return None

    token = cleaned.split()[0]
    if not token:
        return None

    token = token.split("[", maxsplit=1)[0]
    token = _SPECIFIER_PATTERN.split(token, maxsplit=1)[0]

    return token or None


def extract_imports_from_file(file_path: Path) -> Set[str]:
    """Extract all top-level import names from a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Get top-level module name
                module = alias.name.split(".")[0]
                imports.add(module)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # Get top-level module name
                module = node.module.split(".")[0]
                imports.add(module)

    return imports


def get_declared_dependencies() -> Set[str]:
    """Get all dependencies declared in pyproject.toml."""
    pyproject_file = Path("pyproject.toml")
    if not pyproject_file.exists():
        return set()

    data = tomllib.loads(pyproject_file.read_text(encoding="utf-8"))
    project = data.get("project", {})

    dependencies: Set[str] = set()

    for entry in project.get("dependencies", []):
        name = _extract_requirement_name(entry)
        if name:
            dependencies.add(name.replace("-", "_").lower())

    for group in project.get("optional-dependencies", {}).values():
        for entry in group:
            name = _extract_requirement_name(entry)
            if name:
                dependencies.add(name.replace("-", "_").lower())

    return dependencies


def test_all_test_imports_are_declared() -> None:
    """Ensure all imports in test files are either stdlib, declared dependencies, or project modules."""
    test_dir = Path("tests")
    if not test_dir.exists():
        pytest.skip("No tests directory found")

    all_imports = set()
    test_files_checked = 0

    # Scan all test files
    for test_file in test_dir.rglob("*.py"):
        # Skip __pycache__ and other non-test files
        if "__pycache__" in str(test_file):
            continue

        imports = extract_imports_from_file(test_file)
        all_imports.update(imports)
        test_files_checked += 1

    assert test_files_checked > 0, "No test files found"

    # Get declared dependencies
    declared = get_declared_dependencies()

    # Filter out modules that don't need to be declared
    undeclared = (
        all_imports
        - STDLIB_MODULES
        - TEST_FRAMEWORK_MODULES
        - PROJECT_MODULES
        - declared
    )

    # Some packages have different import names than package names
    # Handle known exceptions
    known_mappings = {
        "yaml": "pyyaml",
        "PIL": "pillow",
        "sklearn": "scikit_learn",
        "cv2": "opencv_python",
    }

    # Remove imports that have known package mappings and are declared
    for import_name, package_name in known_mappings.items():
        if import_name in undeclared and package_name in declared:
            undeclared.discard(import_name)

    if undeclared:
        error_msg = (
            f"The following imports in test files are not declared in pyproject.toml:\n"
            f"{', '.join(sorted(undeclared))}\n\n"
            f"Add these packages to [project.optional-dependencies.dev] so tests can run.\n"
            f"Scanned {test_files_checked} test files.\n\n"
            f"🔧 To automatically fix this, run:\n"
            f"   python scripts/sync_test_dependencies.py --fix\n"
            f"   make lock"
        )
        pytest.fail(error_msg)


def test_lockfile_is_present() -> None:
    """Verify requirements.lock exists and appears to be uv-generated."""
    lock_file = Path("requirements.lock")
    assert lock_file.exists(), "requirements.lock not found"

    lines = lock_file.read_text(encoding="utf-8").splitlines()
    assert lines, "requirements.lock is empty"
    assert lines[0].startswith(
        "# This file was autogenerated by uv"
    ), "requirements.lock should be generated via 'make lock'"


def test_external_tools_are_documented() -> None:
    """Ensure external CLI tools used in tests are documented."""
    # Import from the module path
    from pathlib import Path

    # Add tests directory to path to import test_test_dependencies
    tests_dir = Path(__file__).parent
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))

    from test_test_dependencies import OPTIONAL_CLI_TOOLS, REQUIRED_CLI_TOOLS

    documented_tools = set(REQUIRED_CLI_TOOLS.keys()) | set(OPTIONAL_CLI_TOOLS.keys())

    # Scan test files for subprocess calls
    test_dir = Path("tests")
    if not test_dir.exists():
        pytest.skip("No tests directory found")

    used_commands = set()
    for test_file in test_dir.rglob("*.py"):
        if "__pycache__" in str(test_file):
            continue

        try:
            with open(test_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Look for subprocess.run, subprocess.call, etc.
            if "subprocess." in content or "shutil.which" in content:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    # Look for string literals in subprocess calls
                    if isinstance(node, ast.Str) and isinstance(node.s, str):
                        # Check if it's a common CLI tool
                        for tool in ["node", "npm", "uv", "git", "docker"]:
                            if tool in node.s.lower():
                                used_commands.add(tool)
                    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                        for tool in ["node", "npm", "uv", "git", "docker"]:
                            if tool in node.value.lower():
                                used_commands.add(tool)

        except (SyntaxError, UnicodeDecodeError):
            continue

    # Check if used commands are documented
    undocumented = used_commands - documented_tools

    # git and docker are usually system-level, so we can ignore them
    undocumented.discard("git")
    undocumented.discard("docker")

    if undocumented:
        error_msg = (
            f"The following CLI tools are used in tests but not documented:\n"
            f"{', '.join(sorted(undocumented))}\n\n"
            f"Add them to test_test_dependencies.py in REQUIRED_CLI_TOOLS or OPTIONAL_CLI_TOOLS."
        )
        pytest.fail(error_msg)
