#!/usr/bin/env python3
"""
Automatically sync test dependencies to requirements.txt.

This script scans test files for imports and ensures all discovered
dependencies are declared in requirements.txt. It can be run manually
or triggered by CI when test_dependency_enforcement.py fails.

Usage:
    python scripts/sync_test_dependencies.py           # Dry run (shows what would change)
    python scripts/sync_test_dependencies.py --fix     # Update requirements.txt
    python scripts/sync_test_dependencies.py --verify  # Exit 1 if changes needed
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Set


# Stdlib modules that don't need to be installed (keep in sync with test_dependency_enforcement.py)
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

TEST_FRAMEWORK_MODULES = {
    "pytest",
    "hypothesis",
    "_pytest",
    "pluggy",
}

PROJECT_MODULES = {
    "trend_analysis",
    "trend_portfolio_app",
    "streamlit_app",
    "trend_model",
    "trend",
    "app",
    "tools",
    "scripts",
    "tests",
    "_autofix_diag",
    "gate_summary",
    "restore_branch_snapshots",
    "test_test_dependencies",
}

# Module name to package name mappings for known exceptions
MODULE_TO_PACKAGE = {
    "yaml": "PyYAML",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "pre_commit": "pre-commit",
}


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
                module = alias.name.split(".")[0]
                imports.add(module)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split(".")[0]
                imports.add(module)

    return imports


def get_all_test_imports() -> Set[str]:
    """Get all imports used across all test files."""
    test_dir = Path("tests")
    if not test_dir.exists():
        return set()

    all_imports = set()
    for test_file in test_dir.rglob("*.py"):
        if "__pycache__" in str(test_file):
            continue
        imports = extract_imports_from_file(test_file)
        all_imports.update(imports)

    return all_imports


def get_declared_dependencies() -> tuple[Set[str], list[str]]:
    """Get declared dependencies and the raw requirements lines."""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        return set(), []

    dependencies = set()
    raw_lines = []

    with open(requirements_file, "r") as f:
        for line in f:
            raw_lines.append(line.rstrip("\n"))
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            pkg = (
                line.split("==")[0]
                .split(">=")[0]
                .split("<=")[0]
                .split("~=")[0]
                .split("[")[0]
                .strip()
            )
            if pkg:
                module_name = pkg.replace("-", "_").lower()
                dependencies.add(module_name)
                if "-" in pkg:
                    dependencies.add(pkg.lower())

    return dependencies, raw_lines


def find_missing_dependencies() -> Set[str]:
    """Find imports that are not declared as dependencies."""
    all_imports = get_all_test_imports()
    declared, _ = get_declared_dependencies()

    # Filter out modules that don't need to be declared
    potential_deps = (
        all_imports
        - STDLIB_MODULES
        - TEST_FRAMEWORK_MODULES
        - PROJECT_MODULES
        - declared
    )

    # Handle known module-to-package mappings
    missing = set()
    for import_name in potential_deps:
        # Check if it has a known package mapping and that package is declared
        package_name = MODULE_TO_PACKAGE.get(import_name)
        if package_name:
            package_normalized = package_name.replace("-", "_").lower()
            if package_normalized not in declared:
                missing.add(package_name)
        else:
            # Use the import name as the package name
            missing.add(import_name)

    return missing


def add_dependencies_to_requirements(missing: Set[str], fix: bool = False) -> bool:
    """Add missing dependencies to requirements.txt."""
    if not missing:
        return False

    requirements_file = Path("requirements.txt")
    _, raw_lines = get_declared_dependencies()

    # Find the test section in requirements.txt
    test_section_start = -1
    for i, line in enumerate(raw_lines):
        if "# Test dependencies" in line or "# Testing" in line:
            test_section_start = i
            break

    if test_section_start == -1:
        # No test section found, add one at the end
        if raw_lines and not raw_lines[-1].strip():
            raw_lines = raw_lines[:-1]  # Remove trailing blank line
        raw_lines.append("")
        raw_lines.append("# Test dependencies (auto-discovered)")
        test_section_start = len(raw_lines) - 1

    # Insert missing dependencies after the test section header
    insert_position = test_section_start + 1
    for dep in sorted(missing):
        raw_lines.insert(insert_position, dep)
        insert_position += 1

    if fix:
        with open(requirements_file, "w") as f:
            f.write("\n".join(raw_lines) + "\n")
        return True

    return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sync test dependencies to requirements.txt"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Update requirements.txt with missing dependencies",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Exit with code 1 if changes are needed (for CI)",
    )
    args = parser.parse_args()

    missing = find_missing_dependencies()

    if not missing:
        print("✅ All test dependencies are declared in requirements.txt")
        return 0

    print(f"⚠️  Found {len(missing)} undeclared dependencies:")
    for dep in sorted(missing):
        print(f"  - {dep}")

    if args.fix:
        add_dependencies_to_requirements(missing, fix=True)
        print(f"\n✅ Added {len(missing)} dependencies to requirements.txt")
        print("Please run: uv pip compile pyproject.toml -o requirements.lock")
        return 0
    elif args.verify:
        print("\n❌ Run: python scripts/sync_test_dependencies.py --fix")
        return 1
    else:
        print("\nTo fix, run: python scripts/sync_test_dependencies.py --fix")
        return 0


if __name__ == "__main__":
    sys.exit(main())
