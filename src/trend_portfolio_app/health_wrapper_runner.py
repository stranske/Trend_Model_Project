"""Executable module ensuring src path injection prior to loading health
wrapper.

Some CI environments invoke `python -m trend_portfolio_app.health_wrapper` which
fails if the package is not installed and `src` is not on `sys.path`.
This runner can be targeted instead (update smoke test) to guarantee import.
"""

from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from trend_portfolio_app.health_wrapper import main  # noqa: E402

if __name__ == "__main__":  # pragma: no cover
    main()
