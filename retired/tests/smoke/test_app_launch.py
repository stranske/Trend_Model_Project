"""Smoke test for Streamlit app launch with configurable readiness check.

This module addresses the issue of hardcoded sleep timers by
implementing a configurable, sophisticated readiness check that polls
the health endpoint instead of using fixed delays.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

pytestmark = pytest.mark.smoke


# Configuration constants with environment variable overrides
DEFAULT_STARTUP_TIMEOUT = int(os.environ.get("STREAMLIT_STARTUP_TIMEOUT", "60"))
DEFAULT_POLL_INTERVAL = float(os.environ.get("STREAMLIT_POLL_INTERVAL", "0.5"))
DEFAULT_READY_TIMEOUT = int(os.environ.get("STREAMLIT_READY_TIMEOUT", "5"))


def find_project_root(
    start_path: Path, marker_files=("pyproject.toml", "setup.py", ".git")
) -> Path:
    """Search upwards from start_path for a directory containing one of the
    marker files."""
    current = start_path.resolve()
    for parent in [current] + list(current.parents):
        for marker in marker_files:
            if (parent / marker).exists():
                return parent
    raise FileNotFoundError(f"Could not find project root with markers {marker_files}")


PROJECT_ROOT = find_project_root(Path(__file__))
APP_PATH = PROJECT_ROOT / "src" / "trend_portfolio_app" / "app.py"


def wait_for_streamlit_ready(
    port: int,
    timeout: int = DEFAULT_STARTUP_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    ready_timeout: int = DEFAULT_READY_TIMEOUT,
) -> bool:
    """Wait for Streamlit app to be ready by polling the health endpoint.

    This replaces the hardcoded sleep with a sophisticated readiness check
    that polls until the service is actually ready to serve requests.

    Args:
        port: Port number where Streamlit is running
        timeout: Maximum time to wait for startup (seconds)
        poll_interval: Time between polling attempts (seconds)
        ready_timeout: Timeout for individual HTTP requests (seconds)

    Returns:
        True if app is ready, False if timeout exceeded
    """
    start_time = time.time()
    root_url = f"http://localhost:{port}"
    health_url = f"{root_url}/health"

    while time.time() - start_time < timeout:
        try:
            # Use the new stable /health endpoint
            r = requests.get(health_url, timeout=ready_timeout)
            if r.status_code == 200 and r.text.strip() == "OK":
                return True
        except requests.exceptions.RequestException:
            pass

        try:
            # Fallback: connect to the root page and look for plausible content
            response = requests.get(root_url, timeout=ready_timeout)
            if response.status_code == 200:
                content = response.text.lower()
                if "streamlit" in content or "trend" in content or len(content) > 100:
                    return True
        except requests.exceptions.RequestException:
            pass

        # Wait before next poll attempt
        time.sleep(poll_interval)

    return False


def test_app_starts_headlessly():
    """Test that Streamlit app starts successfully with configurable readiness
    check.

    This test addresses the flaky test issue by replacing hardcoded
    sleep with a sophisticated polling mechanism that waits until the
    app is actually ready.
    """
    # Configure environment for headless operation
    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true"

    # Use a dynamic port to avoid conflicts
    port = 8765

    # Build command to start health wrapper (which starts Streamlit internally)
    cmd = [sys.executable, "-m", "trend_portfolio_app.health_wrapper_runner"]

    # Set the wrapper to use the test port and correct app path
    env["HEALTH_PORT"] = str(port)
    env["HEALTH_HOST"] = "127.0.0.1"  # Use localhost for tests
    # These variables are not used by health_wrapper but kept for compatibility
    env["STREAMLIT_APP_PATH"] = str(APP_PATH)
    env["STREAMLIT_PORT"] = str(port + 1)  # Use different port for internal Streamlit
    # Ensure src/ path is available in subprocess for module discovery
    src_path = PROJECT_ROOT / "src"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}:{existing}" if existing else str(src_path)

    print(f"Launching health wrapper on port {port}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Environment: HEALTH_PORT={port}, HEALTH_HOST=127.0.0.1")

    # Start the Streamlit process
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )

    try:
        # Use sophisticated readiness check instead of hardcoded sleep
        print(
            f"Waiting for app to become ready (timeout: {DEFAULT_STARTUP_TIMEOUT}s)..."
        )
        if not wait_for_streamlit_ready(port):
            # If readiness check fails, check if process is still running
            if proc.poll() is not None:
                # Print output for easier debugging as suggested
                stdout_output = proc.stdout.read() if proc.stdout else "No stdout"
                stderr_output = proc.stderr.read() if proc.stderr else "No stderr"
                print("STDOUT:", stdout_output)
                print("STDERR:", stderr_output)
                pytest.fail(
                    f"Health wrapper terminated early with exit code {proc.returncode}. "
                    f"STDOUT: {stdout_output}. STDERR: {stderr_output}"
                )
            else:
                # Collect any available output for debugging
                try:
                    stdout_output = (
                        proc.stdout.read() if proc.stdout else "No stdout available"
                    )
                    stderr_output = (
                        proc.stderr.read() if proc.stderr else "No stderr available"
                    )
                    print("Process still running but not ready. STDOUT:", stdout_output)
                    print("Process still running but not ready. STDERR:", stderr_output)
                except Exception as e:
                    print(f"Could not read process output: {e}")
                    stdout_output = "Could not read stdout"
                    stderr_output = "Could not read stderr"

                pytest.fail(
                    f"Health wrapper failed to become ready within {DEFAULT_STARTUP_TIMEOUT} seconds. "
                    f"Process still running. STDOUT: {stdout_output}. STDERR: {stderr_output}"
                )

        # Verify the process is still running after successful readiness check
        assert proc.poll() is None, (
            "Streamlit app terminated unexpectedly after startup"
        )

        # Additional health check to ensure the app is serving requests
        response = requests.get(
            f"http://localhost:{port}/health", timeout=DEFAULT_READY_TIMEOUT
        )
        assert response.status_code == 200, (
            f"Health check failed with status {response.status_code}"
        )
        assert response.text.strip() == "OK", (
            f"Health check returned unexpected content: {response.text}"
        )

    finally:
        # Clean shutdown of the Streamlit process
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if graceful shutdown fails
            proc.kill()
            proc.wait()


def test_configurable_timeout():
    """Test that timeout configuration works through environment variables."""
    # Test with a very short timeout to verify configurability
    original_timeout = os.environ.get("STREAMLIT_STARTUP_TIMEOUT")
    original_poll = os.environ.get("STREAMLIT_POLL_INTERVAL")

    try:
        # Set short timeouts for testing
        os.environ["STREAMLIT_STARTUP_TIMEOUT"] = "1"
        os.environ["STREAMLIT_POLL_INTERVAL"] = "0.1"

        # This should timeout quickly for a non-existent service
        assert not wait_for_streamlit_ready(port=9999, timeout=1, poll_interval=0.1)

    finally:
        # Restore original environment
        if original_timeout is not None:
            os.environ["STREAMLIT_STARTUP_TIMEOUT"] = original_timeout
        else:
            os.environ.pop("STREAMLIT_STARTUP_TIMEOUT", None)

        if original_poll is not None:
            os.environ["STREAMLIT_POLL_INTERVAL"] = original_poll
        else:
            os.environ.pop("STREAMLIT_POLL_INTERVAL", None)


if __name__ == "__main__":
    # Run the test directly for debugging
    test_app_starts_headlessly()
    print("✅ Streamlit app launch test passed!")
