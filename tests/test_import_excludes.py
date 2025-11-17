"""Ensure legacy directories remain non-importable in editable installs."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
LEGACY_MODULES = ("Old", "retired")


def _points_to_repo_root(entry: str) -> bool:
    """Return True when *entry* refers to the repository root."""
    if entry == "":
        return True  # empty entry maps to the current working directory (repo root during tests)
    try:
        return Path(entry).resolve() == REPO_ROOT
    except (FileNotFoundError, RuntimeError):
        # Some virtualenv entries rely on symlinks or ephemeral paths; treat them as external.
        return False


@contextmanager
def _editable_sys_path() -> None:
    """Mimic pip's editable install where only ``src`` is exposed."""
    original = list(sys.path)
    cleaned = [entry for entry in original if not _points_to_repo_root(entry)]
    sys.path[:] = [str(SRC_DIR)] + cleaned
    try:
        yield
    finally:
        sys.path[:] = original


@pytest.mark.parametrize("module_name", LEGACY_MODULES)
def test_legacy_modules_not_importable(module_name: str) -> None:
    with _editable_sys_path():
        spec = importlib.util.find_spec(module_name)
    assert spec is None, f"Editable installs should not resolve '{module_name}'."
