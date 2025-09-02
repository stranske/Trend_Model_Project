import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

APP_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "trend_portfolio_app" / "app.py"
)


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
