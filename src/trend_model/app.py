"""Console entry point that launches the Streamlit interface."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[2] / "streamlit_app" / "app.py"


def build_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the Streamlit launcher."""

    parser = argparse.ArgumentParser(
        prog="trend-app",
        description="Launch the Trend Model Streamlit experience.",
    )
    parser.add_argument(
        "streamlit_args",
        nargs=argparse.REMAINDER,
        help="Optional arguments forwarded to `streamlit run`.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Streamlit application via ``streamlit run``."""

    parser = build_parser()
    args = parser.parse_args(argv)

    forwarded = list(args.streamlit_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH)]
    cmd.extend(forwarded)

    result = subprocess.run(cmd, check=False)
    return int(result.returncode)


__all__ = ["main", "build_parser"]

