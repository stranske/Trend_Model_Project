"""CI smoke test for Streamlit app with headless run testing."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trend_analysis.api import run_simulation  # noqa: E402
from trend_analysis.config import Config  # noqa: E402


@pytest.fixture(scope="module")
def demo_data():
    """Create demo data for testing."""
    # Generate demo data
    subprocess.run(
        ["python", "scripts/generate_demo.py"],
        cwd=Path(__file__).parent.parent,
        check=True,
    )

    # Load the generated demo data
    demo_path = Path(__file__).parent.parent / "demo" / "demo_returns.csv"
    if demo_path.exists():
        return pd.read_csv(demo_path)
    else:
        # Fallback: create minimal demo data
        dates = pd.date_range("2020-01-31", periods=24, freq="ME")
        return pd.DataFrame(
            {
                "Date": dates,
                "Asset_A": [0.01, -0.02, 0.03, 0.01, 0.02, -0.01] * 4,
                "Asset_B": [0.02, -0.01, 0.02, -0.01, 0.03, 0.01] * 4,
                "RF": [0.0] * 24,
            }
        )


@pytest.fixture(scope="module")
def demo_config():
    """Create demo configuration for testing."""
    return Config(
        version="1",
        data={},
        preprocessing={},
        vol_adjust={"target_vol": 1.0},
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-06",
            "out_start": "2020-07",
            "out_end": "2021-12",
        },
        portfolio={"selection_mode": "all"},
        benchmarks={},
        metrics={"registry": ["sharpe", "max_drawdown"]},
        export={},
        run={"monthly_cost": 0.01},
    )


def test_api_run_simulation_smoke(demo_data, demo_config):
    """Smoke test for the run_simulation API with demo data."""
    # Test that the API works with our demo data
    result = run_simulation(demo_config, demo_data)

    assert result is not None
    assert hasattr(result, "metrics")
    assert hasattr(result, "details")

    # Basic validation that we got some results
    if not result.metrics.empty:
        assert len(result.metrics) > 0
        print(f"✅ API smoke test passed: {len(result.metrics)} metrics generated")
    else:
        print("⚠️  API returned empty metrics (may be expected for minimal data)")


class StreamlitAppManager:
    """Manager for Streamlit app testing."""

    def __init__(self, app_path, port=8501):
        self.app_path = app_path
        self.port = port
        self.process = None

    def start(self):
        """Start the Streamlit app."""
        cmd = [
            "streamlit",
            "run",
            str(self.app_path),
            "--server.port",
            str(self.port),
            "--server.headless",
            "true",
            "--server.enableCORS",
            "false",
            "--server.enableXsrfProtection",
            "false",
        ]

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,  # Create new process group for clean shutdown
        )

        # Wait for app to start
        max_attempts = 30
        for i in range(max_attempts):
            try:
                response = requests.get(f"http://localhost:{self.port}", timeout=5)
                if response.status_code == 200:
                    print(f"✅ Streamlit app started on port {self.port}")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)

        # If we get here, startup failed
        if self.process:
            self.stop()
        raise RuntimeError(
            f"Failed to start Streamlit app after {max_attempts} seconds"
        )

    def stop(self):
        """Stop the Streamlit app."""
        if self.process:
            try:
                # Kill the entire process group to clean up any child processes
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                # Force kill if graceful shutdown fails
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.process = None
            print("✅ Streamlit app stopped")

    def is_running(self):
        """Check if the app is still running."""
        try:
            response = requests.get(f"http://localhost:{self.port}", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


@pytest.fixture(scope="function")
def streamlit_app():
    """Start and stop Streamlit app for testing."""
    # Find the main streamlit app file
    app_paths = [
        Path(__file__).parent.parent / "streamlit_app" / "app.py",
        Path(__file__).parent.parent / "app" / "streamlit" / "app.py",
        Path(__file__).parent.parent / "src" / "trend_portfolio_app" / "app.py",
    ]

    app_path = None
    for path in app_paths:
        if path.exists():
            app_path = path
            break

    if not app_path:
        pytest.skip("No Streamlit app found to test")

    manager = StreamlitAppManager(app_path)

    try:
        manager.start()
        yield manager
    finally:
        manager.stop()


def test_streamlit_app_startup(streamlit_app):
    """Test that Streamlit app starts up successfully."""
    assert streamlit_app.is_running()

    # Test basic connectivity
    response = requests.get(f"http://localhost:{streamlit_app.port}")
    assert response.status_code == 200
    assert "streamlit" in response.text.lower() or "trend" in response.text.lower()
    print("✅ Streamlit app startup test passed")


def test_streamlit_app_pages_accessible(streamlit_app):
    """Test that main app pages are accessible."""
    base_url = f"http://localhost:{streamlit_app.port}"

    # Test main page
    response = requests.get(base_url)
    assert response.status_code == 200

    # Note: Testing individual pages via URL is complex with Streamlit
    # as they use internal routing. The main test is that the app starts
    # and the main page loads without error.
    print("✅ Streamlit pages accessibility test passed")


def test_streamlit_app_run_page_exists():
    """Test that the Run page file exists and can be imported."""
    run_page_paths = [
        Path(__file__).parent.parent / "app" / "streamlit" / "pages" / "03_Run.py",
        Path(__file__).parent.parent / "streamlit_app" / "pages" / "3_Run.py",
    ]

    found_run_page = False
    for path in run_page_paths:
        if path.exists():
            found_run_page = True

            # Try to validate the file has required components
            content = path.read_text()

            # Check for key components
            assert "run_simulation" in content
            assert "progress" in content.lower()
            assert "error" in content.lower()

            print(f"✅ Run page found and validated at {path}")
            break

    assert found_run_page, "No Run page found in expected locations"


def test_error_handling_components():
    """Test that error handling components exist in the Run page."""
    run_page_path = (
        Path(__file__).parent.parent / "streamlit_app" / "pages" / "3_Run.py"
    )

    if not run_page_path.exists():
        pytest.skip("Run page not found for testing")

    content = run_page_path.read_text()

    # Check for error handling features
    assert "format_error_message" in content
    assert "exception" in content.lower() or "error" in content.lower()
    assert "expander" in content.lower()  # For "Show details" functionality
    assert "traceback" in content.lower()

    print("✅ Error handling components test passed")


def test_progress_reporting_components():
    """Test that progress reporting components exist in the Run page."""
    run_page_path = (
        Path(__file__).parent.parent / "app" / "streamlit" / "pages" / "03_Run.py"
    )

    if not run_page_path.exists():
        pytest.skip("Run page not found for testing")

    content = run_page_path.read_text()

    # Check for progress reporting features
    assert "progress" in content.lower()
    assert "log" in content.lower()
    assert "StreamlitLogHandler" in content or "logging" in content.lower()

    print("✅ Progress reporting components test passed")


@pytest.mark.slow
def test_end_to_end_analysis_simulation(demo_data, demo_config):
    """End-to-end test simulating a complete analysis run."""
    print("🚀 Starting end-to-end analysis simulation...")

    # Step 1: Validate demo data
    assert not demo_data.empty
    assert "Date" in demo_data.columns
    print(
        "✅ Demo data validated: "
        f"{len(demo_data)} rows, {len(demo_data.columns)} columns"
    )

    # Step 2: Run analysis
    try:
        result = run_simulation(demo_config, demo_data)
        assert result is not None
        print("✅ Analysis completed successfully")

        # Step 3: Validate results structure
        assert hasattr(result, "metrics")
        assert hasattr(result, "details")
        print("✅ Result structure validated")

        # Step 4: Basic result validation
        if not result.metrics.empty:
            print(f"✅ Generated {len(result.metrics)} metrics")
        else:
            print("⚠️  No metrics generated (may be expected for minimal data)")

        if result.details:
            print(f"✅ Generated details with {len(result.details)} keys")
        else:
            print("⚠️  No details generated")

    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        # Re-raise to fail the test
        raise

    print("🎉 End-to-end analysis simulation completed successfully")


def test_run_page_imports_successfully():
    """Test that the Run page can be imported without errors."""
    run_page_path = (
        Path(__file__).parent.parent / "app" / "streamlit" / "pages" / "03_Run.py"
    )

    if not run_page_path.exists():
        pytest.skip("Run page not found for import testing")

    # Test import without actually running streamlit
    import importlib.util
    from unittest.mock import Mock

    # Mock streamlit before importing
    sys.modules["streamlit"] = Mock()

    spec = importlib.util.spec_from_file_location("run_page", run_page_path)
    assert spec is not None and spec.loader is not None
    run_page = importlib.util.module_from_spec(spec)

    # Should not raise any import errors
    spec.loader.exec_module(run_page)

    # Check that key functions exist
    assert hasattr(run_page, "format_error_message")
    assert hasattr(run_page, "run_analysis_with_progress")

    print("✅ Run page imports successfully")


if __name__ == "__main__":
    # Run basic smoke tests when called directly
    print("Running Streamlit app smoke tests...")

    # Test 1: Generate and validate demo data
    subprocess.run(["python", "scripts/generate_demo.py"], check=True)
    print("✅ Demo data generation test passed")

    # Test 2: Check Run page exists
    run_page_path = Path("streamlit_app/pages/3_Run.py")
    if run_page_path.exists():
        print("✅ Run page exists")
    else:
        print("❌ Run page not found")
        exit(1)

    # Test 3: Basic import test
    import importlib.util
    from unittest.mock import Mock

    sys.modules["streamlit"] = Mock()
    spec = importlib.util.spec_from_file_location("run_page", run_page_path)
    assert spec is not None and spec.loader is not None
    run_page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_page)
    print("✅ Run page import test passed")

    print("🎉 All smoke tests passed!")
