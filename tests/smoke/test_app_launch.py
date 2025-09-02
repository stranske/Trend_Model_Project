import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

def find_project_root(start_path: Path, marker_files=("pyproject.toml", "setup.py", ".git")) -> Path:
    """Search upwards from start_path for a directory containing one of the marker files."""
    current = start_path.resolve()
    for parent in [current] + list(current.parents):
        for marker in marker_files:
            if (parent / marker).exists():
                return parent
    raise FileNotFoundError(f"Could not find project root with markers {marker_files}")

PROJECT_ROOT = find_project_root(Path(__file__))
APP_PATH = PROJECT_ROOT / "src" / "trend_portfolio_app" / "app.py"
def test_app_starts_headlessly():
    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.headless=true",
        "--server.port=8765",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        env=env,
    )
    try:
        time.sleep(5)
        assert proc.poll() is None, "Streamlit app terminated early"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
