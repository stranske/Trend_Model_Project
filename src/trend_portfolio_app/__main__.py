"""Entry point for running trend_portfolio_app as a module.

This allows the package to be run with:
- python -m trend_portfolio_app.health_wrapper (for health service)
- python -m trend_portfolio_app (defaults to Streamlit app)
"""

import sys
from pathlib import Path


def main() -> None:
    """Main entry point for trend_portfolio_app module."""
    # Add src to path to ensure imports work correctly
    repo_root = Path(__file__).parent.parent.parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # This main() is only called for 'python -m trend_portfolio_app'
    # For 'python -m trend_portfolio_app.health_wrapper', the health_wrapper.py
    # module is executed directly and this main() is not called.
    print("Starting Streamlit app...")
    print("For health service, use: python -m trend_portfolio_app.health_wrapper")

    # Import and run streamlit app
    import streamlit.web.cli as stcli

    app_path = Path(__file__).parent / "app.py"
    sys.argv = ["streamlit", "run", str(app_path)]
    stcli.main()


if __name__ == "__main__":
    main()
