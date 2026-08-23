"""Sentinel policy shared by the public and inner demo runners."""

from __future__ import annotations

from pathlib import Path


def consume_fast_sentinel(sentinel: Path, *, release_verify: bool) -> bool:
    """Return whether fast mode should exit, refusing it during release verification."""

    if not sentinel.exists():
        return False
    if release_verify:
        raise SystemExit("release verification refused to skip the real demo")
    try:
        sentinel.unlink()
    except OSError:
        pass
    return True
