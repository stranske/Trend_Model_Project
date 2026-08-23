#!/usr/bin/env python
"""Quick validation of the _run_analysis removal changes."""

import sys
import ast
from pathlib import Path

def test_api_module():
    """Test that _run_analysis is removed from api.py."""
    api_path = Path("src/trend_analysis/api.py")
    tree = ast.parse(api_path.read_text(encoding="utf-8"))
    definitions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    
    if "_run_analysis" in definitions:
        print("FAIL: _run_analysis still defined in api.py")
        return False
    print("PASS: _run_analysis removed from api.py")
    return True

def test_signal_spec_preparation():
    """Test that signal_spec preparation is in run_simulation."""
    api_path = Path("src/trend_analysis/api.py")
    content = api_path.read_text(encoding="utf-8")
    
    # Check that signal_spec is built before dispatch
    if "signal_spec = None" not in content:
        print("FAIL: signal_spec initialization not found")
        return False
    if "_build_trend_spec" not in content:
        print("FAIL: _build_trend_spec call not found")
        return False
    if "_run_analysis_with_diagnostics" not in content:
        print("FAIL: _run_analysis_with_diagnostics call not found")
        return False
    print("PASS: signal_spec preparation found in run_simulation")
    return True

def test_no_api_run_analysis_imports():
    """Test that no code imports api._run_analysis."""
    repo_root = Path(".")
    retired_api_names = {"_run_analysis"}
    offenders = []
    
    for root_name in ("src", "tests", "scripts", "streamlit_app"):
        root_path = repo_root / root_name
        if not root_path.exists():
            continue
        for source_path in root_path.rglob("*.py"):
            if source_path.name == "test_legacy_surface_absence.py":
                continue  # Skip the test file itself
            try:
                source_tree = ast.parse(source_path.read_text(encoding="utf-8"))
                for node in ast.walk(source_tree):
                    if (
                        isinstance(node, ast.ImportFrom)
                        and node.module == "trend_analysis.api"
                        and any(alias.name in retired_api_names for alias in node.names)
                    ):
                        names = sorted(
                            alias.name for alias in node.names if alias.name in retired_api_names
                        )
                        offenders.append(
                            f"{source_path.relative_to(repo_root).as_posix()}: imports {names}"
                        )
                    if (
                        isinstance(node, ast.Attribute)
                        and node.attr in retired_api_names
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "api"
                    ):
                        offenders.append(
                            f"{source_path.relative_to(repo_root).as_posix()}: api.{node.attr}"
                        )
            except Exception as e:
                print(f"Warning: Could not parse {source_path}: {e}")
    
    if offenders:
        print(f"FAIL: Found {len(offenders)} references to api._run_analysis:")
        for o in offenders:
            print(f"  - {o}")
        return False
    print("PASS: No references to api._run_analysis found")
    return True

if __name__ == "__main__":
    tests = [
        test_api_module,
        test_signal_spec_preparation,
        test_no_api_run_analysis_imports,
    ]
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"ERROR in {test.__name__}: {e}")
            results.append(False)
    
    if all(results):
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)
