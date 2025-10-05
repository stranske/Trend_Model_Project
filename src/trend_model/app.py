"""Console entry point for launching the Streamlit application."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

APP_PATH = Path(__file__).resolve().parents[2] / "streamlit_app" / "app.py"


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the Trend Model Streamlit application."""

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
