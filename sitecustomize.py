"""Test helper to ensure the repository's src/ is on sys.path early.

This allows subprocess `python -m trend_portfolio_app.health_wrapper` used in
smoke tests to resolve the package without needing editable installs or
explicit PYTHONPATH mangling inside each test.  Python automatically imports
`sitecustomize` if it is present on the import path during interpreter start.
The file is lightweight and safe in production environments.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if src_dir.exists():  # pragma: no cover - trivial branch
        s = str(src_dir)
        if s not in sys.path:
            sys.path.insert(0, s)
except Exception:  # pragma: no cover - defensive
    # Silently ignore; test environment will surface import errors if any
    pass
