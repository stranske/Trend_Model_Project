"""Console entry point for launching the Streamlit application."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence
import os

def find_app_path(filename="streamlit_app/app.py", start_path=None, max_depth=5):
    """Search upwards from start_path for the given filename, up to max_depth levels."""
    if start_path is None:
        start_path = Path(__file__).resolve().parent
    for _ in range(max_depth):
        candidate = start_path / filename
        if candidate.exists():
            return candidate
        if start_path.parent == start_path:
            break
        start_path = start_path.parent
    raise FileNotFoundError(f"Could not find {filename} upwards from {Path(__file__).resolve().parent}")

APP_PATH = find_app_path()
def main(argv: Sequence[str] | None = None) -> int:
    """Launch the Trend Model Streamlit application.

    Parameters
    ----------
    argv:
        Optional extra arguments to forward to ``streamlit run``. When ``None``
        (the default) the function forwards any command line arguments supplied
        to the wrapper executable.

    Returns
    -------
    int
        The exit code returned by the ``streamlit`` process. A return value of
        ``127`` indicates that the ``streamlit`` executable could not be
        located on ``PATH``.
    """

    args = list(argv) if argv is not None else list(sys.argv[1:])
    command = ["streamlit", "run", str(APP_PATH), *args]
    try:
        result = subprocess.run(command, check=False)
    except FileNotFoundError:  # pragma: no cover - defensive guard
        print(
            "Error: the 'streamlit' executable was not found. Install the optional 'app' extra.",
            file=sys.stderr,
        )
        return 127
    return result.returncode


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    raise SystemExit(main())
