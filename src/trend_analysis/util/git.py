"""Small git metadata helpers shared by export/reporting code."""

from __future__ import annotations

import subprocess
from typing import Any, cast


def git_hash(subprocess_module: Any = subprocess) -> str:
    """Return the current git commit hash, or ``unknown`` outside a checkout."""

    try:
        return cast(
            str,
            subprocess_module.check_output(
                ["git", "rev-parse", "HEAD"], encoding="utf-8", shell=False
            ).strip(),
        )
    except (subprocess_module.CalledProcessError, FileNotFoundError):
        return "unknown"
