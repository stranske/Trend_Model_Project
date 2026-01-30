from __future__ import annotations

from pathlib import Path

from tests.path_utils import is_relative_to


def test_is_relative_to_true(tmp_path: Path) -> None:
    target = tmp_path / "child" / "file.txt"
    assert is_relative_to(target, tmp_path)


def test_is_relative_to_false(tmp_path: Path) -> None:
    target = tmp_path / "child" / "file.txt"
    other = tmp_path / "other"
    assert not is_relative_to(target, other)
