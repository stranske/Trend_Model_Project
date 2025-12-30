"""Test that streamlit app no longer contains experimental FastAPI code."""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))


def test_streamlit_app_no_experimental_fastapi():
    """Test that streamlit_app/app.py no longer contains experimental FastAPI
    code."""
    streamlit_app_path = repo_root / "streamlit_app" / "app.py"

    # Read the file content
    with open(streamlit_app_path) as f:
        content = f.read()

    # Should not contain experimental FastAPI imports or app creation
    assert "from fastapi import" not in content, "FastAPI imports should be removed"
    assert "FastAPI()" not in content, "FastAPI app creation should be removed"
    assert "@app.get" not in content, "FastAPI route decorators should be removed"
    assert "async def health" not in content, "Async health function should be removed"

    # Should still be a valid Streamlit app
    assert "import streamlit as st" in content, "Should still import Streamlit"
    assert "st.set_page_config" in content, "Should still configure Streamlit"


def test_health_wrapper_has_specific_exceptions():
    """Test that health_wrapper.py uses specific exception types."""
    from trend_portfolio_app import health_wrapper

    # Read the source file to verify exception handling
    health_wrapper_path = Path(health_wrapper.__file__)
    with open(health_wrapper_path) as f:
        content = f.read()

    # Should use specific exception types instead of broad Exception
    assert (
        "(ImportError, ModuleNotFoundError)" in content
    ), "Should catch specific import exceptions"

    # Check for specific app creation exceptions (may be formatted across multiple lines)
    assert (
        "ImportError," in content and "AttributeError," in content and "TypeError," in content
    ), "Should catch specific app creation exceptions"

    # Should not use bare Exception (except in comments)
    lines = content.split("\n")
    code_lines = [
        line for line in lines if not line.strip().startswith("#") and "except Exception" in line
    ]
    assert len(code_lines) == 0, f"Should not use bare Exception: {code_lines}"


if __name__ == "__main__":
    test_streamlit_app_no_experimental_fastapi()
    test_health_wrapper_has_specific_exceptions()
    print("✅ All experimental API removal tests passed!")
