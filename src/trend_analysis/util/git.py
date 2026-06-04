"""Small git metadata helpers shared by export/reporting code."""

from __future__ import annotations

import subprocess


def git_hash() -> str:
    """Return the current git commit hash, or ``unknown`` outside a checkout."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], encoding="utf-8", shell=False
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
