"""Tests for :mod:`trend_analysis.util.hash` helper functions."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from trend_analysis.util import hash as hash_utils


def test_sha256_text_and_bytes_consistency() -> None:
    """Ensure text and byte helpers return the same digest."""

    text = "trend-analysis"
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()

    assert hash_utils.sha256_text(text) == expected
    assert hash_utils.sha256_bytes(text.encode("utf-8")) == expected


def test_sha256_file(tmp_path: Path) -> None:
    """Verify files are hashed using the chunked reader."""

    # Write content large enough to require multiple iterations of the reader.
    content = ("data-line-" * 1024).encode("utf-8")
    file_path = tmp_path / "sample.txt"
    file_path.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    assert hash_utils.sha256_file(file_path) == expected


def test_sha256_config_is_order_independent() -> None:
    """Config hashing should be deterministic regardless of key order."""

    cfg_a = {"alpha": 1, "beta": 2, "nested": {"x": True, "y": False}}
    cfg_b = {"nested": {"y": False, "x": True}, "beta": 2, "alpha": 1}

    assert hash_utils.sha256_config(cfg_a) == hash_utils.sha256_config(cfg_b)


@pytest.mark.parametrize(
    "value",
    [b"", b"single chunk", b"multiple" * 500],
)
def test_sha256_bytes_matches_stdlib(value: bytes) -> None:
    """The raw byte helper should mirror ``hashlib.sha256``."""

    assert hash_utils.sha256_bytes(value) == hashlib.sha256(value).hexdigest()

